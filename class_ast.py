from enum import Enum

# HOMEWORK: For each AST node you will need
# to write a "three_addr_code" function which
# writes the three address code instruction
# for each node.

# HOMEWORK: For each AST node you should also
# write a "linearize" function which provides
# a list of 3 address instructions for the node
# and its descendants.

# Hint: you may want to utilize the class hierarchy
# to avoid redundant code.

# enum for data types in ClassIeR
class Type(Enum):
    INT = 1
    FLOAT = 2

# base class for an AST node. Each node
# has a type and a VR
class ASTNode():
    def __init__(self) -> None:
        self.node_type = None
        self.vr = None

# AST leaf nodes
class ASTLeafNode(ASTNode):
    def __init__(self, value: str) -> None:
        self.value = value
        super().__init__()

    def linearize(self) -> list[str]:
       return [f"{self.vr} = {self.value};"]

######
# A number leaf node

# The value passed in should be a number
# (probably as a string).

# HOMEWORK: Determine if the number is a float
# or int and set the type
######
class ASTNumNode(ASTLeafNode):
    def __init__(self, value: str) -> None:        
        super().__init__(value)
        # Check if the value represents a float
        if '.' in value:
            self.node_type = Type.FLOAT
        else:
            # Treat as INT if no decimal
            self.node_type = Type.INT

    def linearize(self) -> list[str]:
        if self.node_type == Type.FLOAT:
            # Emit float version of load instruction
            return [f"{self.vr} = float2vr({self.value});"]
        else:
            # Emit int version
            return [f"{self.vr} = int2vr({self.value});"]
######
# A program variable leaf node

# The value passed in should be an id name
# eventually it should be the new name generated
# by the symbol table to handle scopes.

# When you create this node, you will also need
# to provide its data type
######
class ASTVarIDNode(ASTLeafNode):
    def __init__(self, value: str, value_type) -> None:
        super().__init__(value)
        self.node_type = value_type  # Set type from symbol table

    def linearize(self) -> list[str]:
        # Load value from variable into virtual register
        return [f"{self.vr} = {self.value};"]

######
# An IO leaf node

# The value passed in should be an id name.
# Because it is an IO node, you do not need
# to get a new name for it.

# When you create this node, you will also need
# to provide its data type. It is recorded in
# the symbol table
######
class ASTIOIDNode(ASTLeafNode):
    def __init__(self, value: str, value_type) -> None:
        super().__init__(value)
        self.node_type = value_type  # Set type from symbol table

    def linearize(self) -> list[str]:
        if self.node_type == Type.INT:
            # Convert input int to virtual register
            return [f"{self.vr} = int2vr({self.value});"]
        else:
            return [f"{self.vr} = float2vr({self.value});"]

######
# Binary operation AST Nodes

# These nodes require their left and right children to be
# provided on creation
######
class ASTBinOpNode(ASTNode):
    def __init__(self, l_child, r_child) -> None:
        # Set the node type to be the same as the left child
        self.l_child = l_child
        self.r_child = r_child
        super().__init__()

    def linearizeOP(self, opi: str, opf: str) -> list[str]:
        # Generate the linearized code for the binary operation
        # based on the types of the children
        ast = self.l_child.linearize() + self.r_child.linearize()
        if self.node_type == Type.INT: # Set the operation based on the type
            op = opi # Integer operation
        else:
            op = opf # Float operation
        # Generate the instruction for the operation
        # and store the result in the virtual register
        ast.append(f"{self.vr} = {op}({self.l_child.vr},{self.r_child.vr});")
        return ast 

class ASTPlusNode(ASTBinOpNode):
    def __init__(self, l_child, r_child) -> None:
        super().__init__(l_child,r_child)
    
    def linearize(self) -> list[str]:
        return self.linearizeOP("addi", "addf")

class ASTMultNode(ASTBinOpNode):
    def __init__(self, l_child, r_child) -> None:
        super().__init__(l_child,r_child) 
    def linearize(self) -> list[str]:
        return self.linearizeOP("multi", "multf")

class ASTMinusNode(ASTBinOpNode):
    def __init__(self, l_child, r_child) -> None:
        super().__init__(l_child,r_child)
    def linearize(self) -> list[str]:
        return self.linearizeOP("subi", "subf")

class ASTDivNode(ASTBinOpNode):
    def __init__(self, l_child, r_child) ->None:
        super().__init__(l_child,r_child)
    def linearize(self) -> list[str]:
        return self.linearizeOP("divi", "divf")

######
# Special BinOp nodes for comparisons

# These operations always return an int value
# (as an untyped register):
# 0 for false and 1 for true.

# Because of this, their node type is always
# an int. However, the operations (eq and lt)
# still need to be typed depending
# on their inputs. If their children are floats
# then you need to use eqf, ltf, etc.
######
class ASTEqNode(ASTBinOpNode):
    def __init__(self, l_child, r_child) ->None:
        self.node_type = Type.INT
        super().__init__(l_child,r_child)
    def linearize(self) -> list[str]:
        ast = self.l_child.linearize() + self.r_child.linearize()
        if self.l_child.node_type == Type.FLOAT or self.r_child.node_type == Type.FLOAT:
            op = "eqf"
        else:
            op = "eqi"
        ast.append(f"{self.vr} = {op}({self.l_child.vr}, {self.r_child.vr});")
        return ast

class ASTLtNode(ASTBinOpNode):
    def __init__(self, l_child, r_child: ASTNode) -> None:
        self.node_type = Type.INT
        super().__init__(l_child,r_child)
    def linearize(self) -> list[str]:
        ast = self.l_child.linearize() + self.r_child.linearize()
        if self.l_child.node_type == Type.FLOAT or self.r_child.node_type == Type.FLOAT:
            op = "ltf"
        else:
            op = "lti"
        ast.append(f"{self.vr} = {op}({self.l_child.vr}, {self.r_child.vr});")
        return ast

######
# Unary operation AST Nodes

# The only operations here are converting
# the bits in a virtual register to another
# virtual register of a different type,
# i.e. corresponding to the CLASSIeR instructions:
# vr_int2float and vr_float2int
######
class ASTUnOpNode(ASTNode):
    def __init__(self, child) -> None:
        self.child = child
        super().__init__()
    def linearizeOP(self, op: str) -> list[str]:
        ast = self.child.linearize()
        ast.append(f"{self.vr} = {op}({self.child.vr});")
        return ast
        
class ASTIntToFloatNode(ASTUnOpNode):
    def __init__(self, child) -> None:
        super().__init__(child)
    def linearize(self) -> list[str]:
        return self.linearizeOP("vr_int2float")

class ASTFloatToIntNode(ASTUnOpNode):
    def __init__(self, child) -> None:
        super().__init__(child)
    def linearize(self) -> list[str]:
        return self.linearizeOP("vr_float2int")

