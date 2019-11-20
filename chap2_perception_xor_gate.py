import chap2_perception_and_gate
import chap2_perception_nand_gate
import chap2_perception_or_gate

def XOR(x1,x2):
    tmp1 = chap2_perception_nand_gate.NAND(x1,x2)
    tmp2 = chap2_perception_or_gate.OR(x1,x2)
    tmp3 = chap2_perception_and_gate.AND(tmp1,tmp2)
    return tmp3
print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))






