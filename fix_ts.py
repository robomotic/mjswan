import re
with open("/home/priamai/mjswan/examples/demo/HoldOnCommand.ts", "r") as f:
    text = f.read()

text = re.sub(r'this\.context\.mjModel\.qpos0\[adr\]', r'this.context.mjModel!.qpos0[adr]', text)

with open("/home/priamai/mjswan/examples/demo/HoldOnCommand.ts", "w") as f:
    f.write(text)
