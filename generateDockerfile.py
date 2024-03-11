from sys import argv

template = 'Dockerfile.template'
output = 'Dockerfile'
image = argv[1]


with open(template, 'r') as f:
    lines = f.read().splitlines()

new_lines = []
for line in lines:
    if line == 'FROM':
        new_lines.append('FROM {}'.format(image))
    else:
        new_lines.append(line)

with open(output, 'w') as f:
    f.write('\n'.join(new_lines)+'\n')
