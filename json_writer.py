import json
def write(data, filename, columns = None):
    if columns:
        data = [columns] + data

    num_cols = max(len(row) for row in data)

    col_widths = [0] * num_cols
    for row in data:
        for i, val in enumerate(row):
            # Convert nested lists to string for width calculation
            s = json.dumps(val)
            col_widths[i] = max(col_widths[i], len(s))

    with open(filename, "w") as f:
        f.write("[\n")
        for idx,row in enumerate(data):
            row_strs = []
            for i, val in enumerate(row):
                s = json.dumps(val)
                # Right-align numbers, left-align everything else
                if isinstance(val, (int, float)):
                    s = f"{s:>{col_widths[i]}}"
                else:
                    s = f"{s:<{col_widths[i]}}"
                row_strs.append(s)
            f.write(f"    [{', '.join(row_strs)}]{',' if idx < len(data) - 1 else ''}\n")
        f.write("]\n")

def read(filename, columns = None):
    with open(filename, "r") as f:
        data = json.load(f)
    
    if columns is not None:
        return data[1:]
    return data

