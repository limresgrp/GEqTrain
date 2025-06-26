from e3nn import o3
from collections import defaultdict


SCALAR = o3.Irrep("0e")
PSEUDO_SCALAR = o3.Irrep("0o")

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

def complete_parities(irreps_in: o3.Irreps) -> o3.Irreps:
    """
    For each l in an Irreps, add the missing parity if it's not present.

    The multiplicity of the new parity is copied from the existing parity
    of the same l. If both parities already exist for a given l, no
    change is made for that l.

    Example:
        "8x0e + 8x1o + 8x1e" -> "8x0e + 8x0o + 8x1e + 8x1o"
        "3x1o + 2x2e" -> "3x1e + 3x1o + 2x2e + 2x2o"
    """
    # 1. Group the existing irreps by their l value.
    #    The structure is: {l: {parity: multiplicity}}
    #    where parity is 1 for even ('e') and -1 for odd ('o').
    l_groups = defaultdict(dict)
    for mul, ir in irreps_in:
        l_groups[ir.l][ir.p] = mul

    # 2. Identify which new irreps need to be created.
    irreps_to_add = []
    # Iterate through the l values present in the input
    for l, parities in l_groups.items():
        has_even = 1 in parities
        has_odd = -1 in parities

        # Case 1: Even exists, but odd is missing
        if has_even and not has_odd:
            # Get the multiplicity from the existing even irrep
            mul = parities[1]
            # Create the corresponding odd irrep and add it to our list
            irreps_to_add.append((mul, o3.Irrep(l, p=-1)))
        
        # Case 2: Odd exists, but even is missing
        elif has_odd and not has_even:
            # Get the multiplicity from the existing odd irrep
            mul = parities[-1]
            # Create the corresponding even irrep and add it to our list
            irreps_to_add.append((mul, o3.Irrep(l, p=1)))
        
        # Case 3: Both or neither exist. Do nothing.

    # If no irreps were added, we can return the original object
    if not irreps_to_add:
        return irreps_in

    # 3. Combine the original irreps with the new ones.
    # The `+` operator concatenates them, and then we sort to get the
    # canonical representation.
    new_irreps_to_add = o3.Irreps(irreps_to_add)
    final_irreps = (irreps_in + new_irreps_to_add).sort().irreps

    return final_irreps