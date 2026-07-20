import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import re
import shlex
from pathlib import Path
import torch

# --- Import the conversion utility ---
try:
    from cartesian_to_spherical import convert_cartesian_to_spherical
except ImportError:
    print("Error: Could not import convert_cartesian_to_spherical. Make sure 'cartesian_to_spherical.py' is in the Python path.")
    exit()

# Mapping of common element symbols to atomic numbers (assuming it's complete for your data)
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36
    # Add more if needed
}

# Mapping of element symbols to their (row, column) in the periodic table (optional, keep if used)
PERIODIC_TABLE_INFO = {
    'H': (1, 1), 'He': (1, 18), 'Li': (2, 1), 'Be': (2, 2), 'B': (2, 13), 'C': (2, 14),
    'N': (2, 15), 'O': (2, 16), 'F': (2, 17), 'Ne': (2, 18), 'Na': (3, 1), 'Mg': (3, 2),
    'Al': (3, 13), 'Si': (3, 14), 'P': (3, 15), 'S': (3, 16), 'Cl': (3, 17), 'Ar': (3, 18),
    'K': (4, 1), 'Ca': (4, 2), 'Sc': (4, 3), 'Ti': (4, 4), 'V': (4, 5), 'Cr': (4, 6),
    'Mn': (4, 7), 'Fe': (4, 8), 'Co': (4, 9), 'Ni': (4, 10), 'Cu': (4, 11), 'Zn': (4, 12),
    'Ga': (4, 13), 'Ge': (4, 14), 'As': (4, 15), 'Se': (4, 16), 'Br': (4, 17), 'Kr': (4, 18)
    # Add more if needed
}

HEADER_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CS_ISO_PROPERTY_ALIASES = ("cs_iso", "CS_total", "CS_iso", "cs_total")


def parse_atom_type_filter(values):
    """Parse element symbols from repeated, comma-separated, or whitespace-separated CLI values."""
    if not values:
        return None

    symbols = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            symbol = part.strip()
            if symbol:
                symbols.append(symbol)

    if not symbols:
        return None

    unknown = [symbol for symbol in symbols if symbol not in ATOMIC_NUMBERS]
    if unknown:
        raise ValueError(
            f"Unknown element symbol(s) in --target-atom-types: {unknown}. "
            f"Known symbols: {sorted(ATOMIC_NUMBERS)}"
        )

    # Preserve user order while removing duplicates.
    return list(dict.fromkeys(symbols))


def apply_target_atom_type_filter(mol_dict, target_atom_type_symbols):
    """Set chemical-shift targets to NaN for atoms outside the requested element set."""
    if target_atom_type_symbols is None:
        return

    allowed_atomic_numbers = np.array(
        [ATOMIC_NUMBERS[symbol] for symbol in target_atom_type_symbols],
        dtype=mol_dict['atom_types'].dtype,
    )
    excluded_atoms = ~np.isin(mol_dict['atom_types'], allowed_atomic_numbers)
    if not np.any(excluded_atoms):
        return

    for key in ("cs_iso", "cs_tensor_spherical"):
        if key in mol_dict:
            target = mol_dict[key].astype(np.float32, copy=True)
            target[excluded_atoms] = np.nan
            mol_dict[key] = target


def parse_properties(properties_str: str):
    """Parses the 'Properties' string from the XYZ header."""
    props = properties_str.split(':')
    prop_info = {}
    current_col = 0
    for i in range(0, len(props), 3):
        name = props[i]
        try:
            ptype = props[i+1]
            dim = int(props[i+2])
        except IndexError:
            print(f"Warning: Malformed Properties string fragment near '{name}'. Skipping subsequent properties.")
            break

        col_slice = slice(current_col, current_col + dim)

        dtype = str
        if ptype == 'R': dtype = np.float32
        elif ptype == 'L': dtype = bool
        elif ptype == 'I': dtype = np.int_
        elif ptype == 'S': dtype = str # Explicitly handle string type
        else:
             print(f"Warning: Unknown property type '{ptype}' for property '{name}'. Treating as string.")

        prop_info[name] = {"slice": col_slice, "dim": dim, "dtype": dtype}
        current_col += dim
    return prop_info


def _first_raw_property(raw_atom_data, names):
    for name in names:
        if name in raw_atom_data:
            return raw_atom_data[name]
    return None


def _parse_header_value(value):
    if value.upper() in ("T", "F"):
        return value.upper() == "T"

    parts = value.split()
    if not parts:
        return value

    try:
        numeric = [float(part) for part in parts]
    except ValueError:
        return value

    if len(numeric) == 1:
        return numeric[0]
    return np.array(numeric, dtype=np.float32)


def _store_standard_atom_properties(mol_dict, raw_atom_data, target_atom_type_symbols):
    cs_iso = _first_raw_property(raw_atom_data, CS_ISO_PROPERTY_ALIASES)
    if cs_iso is not None:
        mol_dict['cs_iso'] = cs_iso.reshape(mol_dict['num_atoms'], 1).astype(np.float32)

    if 'pos' in raw_atom_data:
        mol_dict['pos'] = raw_atom_data['pos']
    if 'forces' in raw_atom_data:
        mol_dict['forces'] = raw_atom_data['forces']
    if 'center_atoms_mask' in raw_atom_data:
        mol_dict['center_atoms_mask'] = raw_atom_data['center_atoms_mask']

    apply_target_atom_type_filter(mol_dict, target_atom_type_symbols)

def _parse_molecule_block(block):
    """
    Parse a single molecule block extracted from an extxyz file.
    """
    block_idx, num_atoms, header, atom_lines, target_atom_type_symbols = block
    try:
        mol_dict = {'num_atoms': num_atoms}
        properties_str = ""

        header_parts = shlex.split(header)
        for part in header_parts:
            if '=' in part:
                key, value = part.split('=', 1)
                if key == 'Properties':
                    properties_str = value
                elif not HEADER_KEY_RE.match(key):
                    # Some generators put loose annotations like "#=T" or
                    # "10.00000=T" in the comment line. They are not useful
                    # dataset fields and would otherwise become NPZ keys.
                    continue
                else:
                    mol_dict[key] = _parse_header_value(value)

        if not properties_str:
            raise ValueError("Missing 'Properties' definition in header.")

        prop_map = parse_properties(properties_str)

        raw_atom_data = {}
        for key, info in prop_map.items():
            shape = (num_atoms, info['dim']) if info['dim'] > 1 else (num_atoms,)
            if info['dtype'] == str:
                raw_atom_data[key] = np.empty(shape, dtype=object)
            else:
                raw_atom_data[key] = np.zeros(shape, dtype=info['dtype'])

        mol_dict['atom_types'] = np.zeros(num_atoms, dtype=np.int64)
        mol_dict['atom_rows'] = np.zeros(num_atoms, dtype=np.int8)
        mol_dict['atom_cols'] = np.zeros(num_atoms, dtype=np.int8)

        sorted_prop_keys = sorted(prop_map.keys(), key=lambda k: prop_map[k]['slice'].start)
        for i, line in enumerate(atom_lines):
            parts = line.split()
            current_part_idx = 0
            processed_keys = set()

            for key in sorted_prop_keys:
                if key in processed_keys:
                    continue
                info = prop_map[key]
                num_parts_for_key = info['dim']
                end_part_idx = current_part_idx + num_parts_for_key

                if end_part_idx > len(parts):
                    raise ValueError(
                        f"Line {block_idx + i + 3}: Not enough columns for property '{key}'. "
                        f"Expected {num_parts_for_key} value(s), found {len(parts) - current_part_idx}."
                    )

                raw_vals = parts[current_part_idx:end_part_idx]

                if key == 'species':
                    species_symbol = raw_vals[0]
                    raw_atom_data[key][i] = species_symbol
                    mol_dict['atom_types'][i] = ATOMIC_NUMBERS.get(species_symbol, 0)
                    row, col = PERIODIC_TABLE_INFO.get(species_symbol, (0, 0))
                    mol_dict['atom_rows'][i] = row
                    mol_dict['atom_cols'][i] = col
                elif info['dtype'] == bool:
                    raw_atom_data[key][i] = True if raw_vals[0].upper() == 'T' else False
                elif info['dtype'] == str:
                    raw_atom_data[key][i] = " ".join(raw_vals) if info['dim'] > 1 else raw_vals[0]
                else:
                    try:
                        val = np.array(raw_vals, dtype=info['dtype'])
                        if info['dim'] > 1:
                            raw_atom_data[key][i, :] = val
                        else:
                            raw_atom_data[key][i] = val
                    except ValueError as conversion_error:
                        raise ValueError(
                            f"Line {block_idx + i + 3}, Property '{key}': Could not convert "
                            f"'{raw_vals}' to {info['dtype']}. Error: {conversion_error}"
                        )

                processed_keys.add(key)
                current_part_idx = end_part_idx

        if 'cs_tensor' in raw_atom_data:
            cs_tensor_cartesian_flat = raw_atom_data['cs_tensor']
            if cs_tensor_cartesian_flat.shape[-1] != 9:
                raise ValueError(
                    f"Molecule {block_idx}: 'cs_tensor' should have 9 components, "
                    f"found {cs_tensor_cartesian_flat.shape[-1]}"
                )

            cs_tensor_cartesian_3x3_np = cs_tensor_cartesian_flat.reshape(num_atoms, 3, 3)
            cs_tensor_cartesian_3x3_torch = torch.from_numpy(cs_tensor_cartesian_3x3_np).to(torch.float32)

            cs_tensor_spherical_torch = convert_cartesian_to_spherical(cs_tensor_cartesian_3x3_torch)
            mol_dict['cs_tensor_spherical'] = cs_tensor_spherical_torch[:, 1:].numpy()

            if 'cs_iso' in raw_atom_data:
                mol_dict['cs_iso'] = raw_atom_data['cs_iso'].reshape(num_atoms, 1).astype(np.float32)
            else:
                mol_dict['cs_iso'] = (
                    np.trace(cs_tensor_cartesian_3x3_np, axis1=1, axis2=2).reshape(num_atoms, 1) / 3.0
                )

            _store_standard_atom_properties(mol_dict, raw_atom_data, target_atom_type_symbols)
        else:
            _store_standard_atom_properties(mol_dict, raw_atom_data, target_atom_type_symbols)
            if 'cs_iso' not in mol_dict:
                print(f"Warning: Molecule {block_idx} missing 'cs_tensor' and scalar CS target. Skipping shielding processing for this molecule.")

        return mol_dict, None
    except (ValueError, IndexError, KeyError) as e:
        return None, f"Error: A parsing error occurred for molecule starting near line {block_idx + 1}. Skipping molecule. Error: {e}"


def parse_extxyz_file(filepath: Path, workers: int = 1, target_atom_type_symbols=None):
    """
    Reads an extended XYZ file and converts Cartesian CS tensors to spherical form.
    If workers > 1, molecule blocks are parsed in parallel.
    """
    molecules_data = []
    print(f"--- Reading and parsing {filepath.name} ---")
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"Error: Could not read file {filepath}. Reason: {e}")
        return []

    line_idx = 0
    mol_count = 0
    blocks = []
    while line_idx < len(lines):
        try:
            if not lines[line_idx].strip(): # Skip empty lines
                line_idx += 1
                continue
            num_atoms = int(lines[line_idx].strip())
            header = lines[line_idx+1].strip()
            atom_lines = lines[line_idx+2 : line_idx+2 + num_atoms]
            if len(atom_lines) != num_atoms:
                raise ValueError(f"Expected {num_atoms} atom lines, but found {len(atom_lines)}.")
            blocks.append((line_idx, num_atoms, header, atom_lines, target_atom_type_symbols))
            mol_count += 1
            line_idx += num_atoms + 2

        except (ValueError, IndexError, KeyError) as e:
            print(f"Error: A parsing error occurred for molecule starting near line {line_idx+1}. Skipping molecule. Error: {e}")
            # Try to advance past the problematic molecule; heuristic: assume header + num_atoms lines
            try:
                potential_num_atoms = int(lines[line_idx].strip())
                line_idx += potential_num_atoms + 2
            except:
                line_idx += 1 # Advance one line if num_atoms couldn't be read

    if workers > 1 and len(blocks) > 1:
        print(f"Parsing {len(blocks)} molecules with {workers} workers.")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            parsed_blocks = list(executor.map(_parse_molecule_block, blocks))
    else:
        if workers > 1:
            print("Only one molecule block found; using a single worker path.")
        parsed_blocks = [_parse_molecule_block(block) for block in blocks]

    for mol_dict, error in parsed_blocks:
        if error:
            print(error)
            continue
        molecules_data.append(mol_dict)

    print(f"Successfully parsed {len(molecules_data)} molecules.")
    if target_atom_type_symbols is not None:
        total_atoms = sum(mol['num_atoms'] for mol in molecules_data)
        finite_target_atoms = sum(
            int(np.isfinite(mol['cs_iso']).all(axis=-1).sum())
            for mol in molecules_data
            if 'cs_iso' in mol
        )
        print(
            "Target atom-type filter kept chemical-shift targets for "
            f"{finite_target_atoms}/{total_atoms} real atoms "
            f"({', '.join(target_atom_type_symbols)})."
        )
    return molecules_data


def create_and_save_masked_npz(molecules, output_path: Path):
    """
    Creates and saves masked raw arrays to an NPZ file.
    """
    if not molecules:
        print("No data to save. Aborting.")
        return

    print("\n--- Merging and Saving Datasets ---")
    total_mols = len(molecules)
    max_n_atoms = max(m['num_atoms'] for m in molecules)
    print(f"Processing {total_mols} structures, padding up to {max_n_atoms} atoms.")

    # Identify keys to save (atom-level and graph-level)
    all_keys = set(key for mol in molecules for key in mol.keys())
    atom_level_keys = set()
    graph_level_keys = set()

    # Define keys we expect at atom level after processing
    expected_atom_keys = {'pos', 'cs_tensor_spherical', 'cs_iso', 'center_atoms_mask',
                          'forces', 'atom_types', 'atom_rows', 'atom_cols'}

    for key in all_keys:
        if key in expected_atom_keys:
            atom_level_keys.add(key)
        elif key != 'num_atoms': # Exclude num_atoms, it's implicit
            # Check if it looks like an atom-level property based on first molecule
            val = molecules[0].get(key)
            if isinstance(val, np.ndarray) and val.shape[0] == molecules[0]['num_atoms']:
                 atom_level_keys.add(key)
            else:
                 graph_level_keys.add(key)


    save_dict = {}

    # --- Process and mask atom-level data ---
    for key in sorted(list(atom_level_keys)):
        print(f"Processing atom-level property: {key}")
        # Find the first molecule that actually has this key to determine dtype/shape
        first_mol_prop = next((m[key] for m in molecules if key in m), None)
        if first_mol_prop is None:
             print(f"  -> Key '{key}' not found in any molecule, skipping.")
             continue # Skip if key is entirely missing

        # Determine shape for the padded array
        if first_mol_prop.ndim > 1:
            shape = (total_mols, max_n_atoms, first_mol_prop.shape[1])
        else:
            shape = (total_mols, max_n_atoms)

        # Create masked array, initialize data to zeros, mask everything initially
        masked_array = np.ma.masked_all(shape, dtype=first_mol_prop.dtype)
        masked_array.data[...] = 0 # Fill underlying data with zeros

        # Fill with actual data and unmask
        for i, mol_data in enumerate(molecules):
            if key in mol_data:
                n_atoms = mol_data['num_atoms']
                data = mol_data[key]
                if data.ndim > 1:
                    masked_array.data[i, :n_atoms, :] = data
                    masked_array.mask[i, :n_atoms, :] = False
                else:
                    masked_array.data[i, :n_atoms] = data
                    masked_array.mask[i, :n_atoms] = False

        save_dict[key] = masked_array.data
        save_dict[f"{key}__mask__"] = masked_array.mask # Save the boolean mask

    # --- Process and stack graph-level data ---
    for key in sorted(list(graph_level_keys)):
        print(f"Processing graph-level property: {key}")
        data_list = [mol.get(key) for mol in molecules]

        # Check if all items are None (e.g., optional fields completely missing)
        if all(item is None for item in data_list):
            print(f"  -> Key '{key}' is None for all molecules, skipping.")
            continue

        try:
            # Attempt to stack if possible, handling potential type/shape inconsistencies
            if any(isinstance(item, np.ndarray) for item in data_list):
                # Find a default shape/dtype from the first valid ndarray
                first_valid_item = next(item for item in data_list if isinstance(item, np.ndarray))
                default_shape = first_valid_item.shape
                default_dtype = first_valid_item.dtype
                # Create a default value (e.g., NaNs or zeros) based on dtype
                if np.issubdtype(default_dtype, np.floating):
                     default_val = np.full(default_shape, np.nan, dtype=default_dtype)
                elif np.issubdtype(default_dtype, np.integer):
                     default_val = np.full(default_shape, 0, dtype=default_dtype) # Or a specific int marker?
                else: # boolean, object, etc.
                     default_val = np.full(default_shape, None, dtype=object) # Use None for object arrays?

                # Process list, replacing non-ndarrays with default
                processed_list = [item if isinstance(item, np.ndarray) else default_val for item in data_list]

                # Ensure all arrays can be stacked (same shape except first dim)
                if all(arr.shape[1:] == default_shape[1:] for arr in processed_list if isinstance(arr,np.ndarray)):
                     stacked_array = np.stack(processed_list, axis=0)
                     save_dict[key] = stacked_array
                else:
                     print(f"Warning: Cannot stack '{key}' due to inconsistent shapes. Saving as object array.")
                     save_dict[key] = np.array(data_list, dtype=object)

            else: # Not arrays, try simple conversion
                save_dict[key] = np.array(data_list, dtype=object)
                # Try converting to float if possible
                try: save_dict[key] = save_dict[key].astype(np.float32)
                except (ValueError, TypeError): pass # Keep as object if conversion fails

            # Specific handling for Lattice (example)
            if key == 'Lattice' and key in save_dict and isinstance(save_dict[key], np.ndarray) and save_dict[key].shape == (total_mols, 9):
                save_dict[key] = save_dict[key].reshape(total_mols, 3, 3)
                print(f"  -> Reshaped 'Lattice' to {save_dict[key].shape}")

        except Exception as e:
            print(f"Warning: Could not collate graph-level property '{key}'. Reason: {e}. Saving as object array.")
            # Fallback to saving as object array if stacking fails unexpectedly
            save_dict[key] = np.array(data_list, dtype=object)

    # --- Save the NPZ file ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"\n✅ Dataset saved successfully to: {output_path}")
    print(f"Saved fields: {list(save_dict.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert extended XYZ file(s) to masked NPZ datasets. "
            "Targets are written in raw form; GEqTrain applies normalization later "
            "from the training data config."
        )
    )
    parser.add_argument(
        "-i", "--inputs", type=Path, nargs='+', required=True,
        help="One or more input extended XYZ files."
    )
    parser.add_argument(
        "-o", "--outputs", type=Path, nargs='*',
        help="Output path(s) for the NPZ file(s). If not provided, defaults to using the input filenames with a .npz extension. If provided, the number of output files must match the number of input files."
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=1,
        help="Number of worker processes to use when parsing each input file. Default: 1 (single process)."
    )
    parser.add_argument(
        "--target-atom-types", nargs="*", default=None,
        help=(
            "Optional element-symbol filter for chemical-shift targets, e.g. "
            "`--target-atom-types H C` or `--target-atom-types H,C`. "
            "Atoms outside this set keep their positions and atom types but get "
            "NaN cs_iso/cs_tensor_spherical targets so GEqTrain can remove them "
            "as supervised edge centers via NaN-aware filtering."
        ),
    )
    args = parser.parse_args()

    try:
        target_atom_type_symbols = parse_atom_type_filter(args.target_atom_types)
    except ValueError as exc:
        parser.error(str(exc))
        return
    if target_atom_type_symbols is not None:
        print(f"Chemical-shift targets will be kept only for atom types: {target_atom_type_symbols}")

    # Determine output paths
    if not args.outputs:
        output_paths = [p.with_suffix('.npz') for p in args.inputs]
    elif len(args.inputs) != len(args.outputs):
        parser.error("The number of --inputs and --outputs must be the same.")
        return # This is for clarity, parser.error exits
    else:
        output_paths = args.outputs

    # 1. Parse all molecules from all input files
    all_datasets_data = []
    for input_path in args.inputs:
        if not input_path.is_file():
            print(f"Error: Input file not found at {input_path}")
            return
        molecules_data = parse_extxyz_file(
            input_path,
            workers=max(1, args.workers),
            target_atom_type_symbols=target_atom_type_symbols,
        )
        if not molecules_data:
            print(f"Warning: No molecules parsed from {input_path}. It will be skipped.")
        all_datasets_data.append(molecules_data)

    # 2. Save the raw converted datasets. GEqTrain will normalize them later
    # according to the `normalization:` section in the training data config.
    for i, (dataset_data, output_path) in enumerate(zip(all_datasets_data, output_paths)):
        print(f"\n--- Processing dataset {i+1}/{len(all_datasets_data)} for output: {output_path} ---")
        if not dataset_data:
            print("Skipping empty dataset.")
            continue
        create_and_save_masked_npz(dataset_data, output_path)

if __name__ == "__main__":
    main()
