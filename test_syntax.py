import json

with open('g:/My Drive/gpu programming/gpu_last_time/practice_pd.ipynb', 'r') as f:
    nb = json.load(f)

code = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # join lines if it is a list
        if isinstance(cell['source'], list):
            code += "".join(cell['source']) + "\n\n"
        else:
            code += cell['source'] + "\n\n"

print("Compiling code...\n")
compile(code, '<string>', 'exec')
print("Compiled successfully!")
