import pdb
import class_ast as class_ast
from class_ast import *
from typing import Callable, List, Tuple, Optional
from scanner import Lexeme, Token, Scanner
from enum import Enum

class IDType(Enum):
    IO = 1
    VAR = 2

class SymbolTableData:
    def __init__(self, id_type: IDType, data_type: Type, new_name: str) -> None:
        self.id_type = id_type
        self.data_type = data_type
        self.new_name = new_name

    def get_id_type(self) -> IDType:
        return self.id_type

    def get_data_type(self) -> Type:
        return self.data_type

    def get_new_name(self) -> str:
        return self.new_name

class SymbolTableException(Exception):
    def __init__(self, lineno: int, ID: str) -> None:
        message = f"Symbol table error on line: {lineno}\nUndeclared ID: {ID}"
        super().__init__(message)

class NewLabelGenerator:
    def __init__(self) -> None:
        self.counter = 0

    def mk_new_label(self) -> str:
        new_label = "label" + str(self.counter)
        self.counter += 1
        return new_label

class NewNameGenerator:
    def __init__(self) -> None:
        self.counter = 0
        self.new_names = []

    def mk_new_name(self) -> str:
        new_name = "_new_name" + str(self.counter)
        self.counter += 1
        self.new_names.append(new_name)
        return new_name

class VRAllocator:
    def __init__(self) -> None:
        self.counter = 0

    def mk_new_vr(self) -> str:
        vr = "vr" + str(self.counter)
        self.counter += 1
        return vr

    def declare_variables(self) -> List[str]:
        ret = []
        for i in range(self.counter):
            ret.append("virtual_reg%d;"%i)
        return ret

class SymbolTable:
    def __init__(self) -> None:
        self.ht_stack = [dict()]

    def insert(self, ID: str, id_type: IDType, data_type: Type) -> None:
        info = SymbolTableData(id_type, data_type, ID)
        self.ht_stack[-1][ID] = info

    def lookup(self, ID: str) -> Optional[SymbolTableData]:
        for ht in reversed(self.ht_stack):
            if ID in ht:
                return ht[ID]
        return None

    def push_scope(self) -> None:
        self.ht_stack.append(dict())

    def pop_scope(self) -> None:
        self.ht_stack.pop()

class ParserException(Exception):
    def __init__(self, lineno: int, lexeme: Lexeme, tokens: List[Token]) -> None:
        message = f"Parser error on line: {lineno}\nExpected one of: {tokens}\nGot: {lexeme}"
        super().__init__(message)

class Parser:
    def __init__(self, scanner: Scanner) -> None:
        self.scanner = scanner
        self.symbol_table = SymbolTable()
        self.vra = VRAllocator()
        self.nlg = NewLabelGenerator()
        self.nng = NewNameGenerator()
        self.function_name = None
        self.function_args = []

    def parse(self, s: str) -> List[str]:
        self.scanner.input_string(s)
        self.to_match = self.scanner.token()
        p = self.parse_function()
        self.eat(None)
        return p

    def get_token_id(self, l: Lexeme) -> Token:
        if l is None:
            return None
        return l.token

    def eat(self, check: Token) -> None:
        token_id = self.get_token_id(self.to_match)
        if token_id !=check:
            raise ParserException(self.scanner.get_lineno(),self.to_match,[check])
        self.to_match = self.scanner.token()

    def parse_function(self) -> List[str]:
        self.parse_function_header()
        self.eat(Token.LBRACE)
        p = self.parse_statement_list()
        self.eat(Token.RBRACE)
        return p

    def parse_function_header(self) -> None:
        self.eat(Token.VOID)
        function_name = self.to_match.value
        self.eat(Token.ID)
        self.eat(Token.LPAR)
        self.function_name = function_name
        args = self.parse_arg_list()
        self.function_args = args
        self.eat(Token.RPAR)

    def parse_arg_list(self) -> List[Tuple[str, str]]:
        token_id = self.get_token_id(self.to_match)
        if token_id == Tokken.RPAR:
            return
        arg = self.parse_arg()
        token_id = self.get_token_id(self.tomatch)
        if token_id == Token.RPAR:
            return [arg]
        self.eat(Token.COMMA)
        arg_l = self.parse_arg_list()
        return arg_l + [arg]

    def parse_arg(self) -> Tuple[str, str]:
        token_id = self.get_token_id(self.to_match)
        if token_id == Token.FLOAT:
            self.eat(Token.FLOAT)
            data_type = Type.FLOAT
            data_type_str = "float"
        elif token_id == Token.INT:
            self.eat(Token.INT)
            data_type = Type.INT
            data_type_str = "int"
        else:
            raise ParserException(self.scanner.get_lineno(), self.to_match, [Token.INT, Token.FLOAT])
        self.eat(Token.AMP)
        id_name = self.to_match.value
        self.eat(Token.ID)
        self.symbol_table.insert(id_name, IDType.IO, data_type)
        return (id_name, data_type_str)

    def parse_statement_list(self) -> List[str]:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.INT, Token.FLOAT, Token.ID, Token.IF, Token.LBRACE, Token.FOR]:
            code1 = self.parse_statement()
            code2 = self.parse_statement_list()
            return code1 + code2
        if token_id in [Token.RBRACE]:
            return []

    def parse_statement(self) -> List[str]:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.INT, Token.FLOAT]:
            return self.parse_declaration_statement()
        elif token_id == [Token.ID]:
            return self.parse_assignment_statement()
        elif token_id == [Token.IF]:
            return self.parse_if_else_statement()
        elif token_id == [Token.LBRACE]:
            return self.parse_block_statement()
        elif token_id == [Token.FOR]:
            return self.parse_for_statement()
        else:
            raise ParserException(self.scanner.get_lineno(), self.to_match,
                                  [Token.FOR, Token.IF, Token.LBRACE, Token.INT, Token.FLOAT, Token.ID])

    def parse_declaration_statement(self) -> List[str]:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.INT]:
            self.eat(Token.INT)  # Consume 'int'
            id_name = self.to_match.value
            new_name = self.nng.mk_new_name()  # Generate unique name
            data = SymbolTableData(IDType.VAR, Type.INT, new_name)
            self.symbol_table.ht_stack[-1][id_name] = data  # Insert into symbol table
            self.eat(Token.ID)  # Consume identifier
            self.eat(Token.SEMI)  # Consume semicolon
            return []
        if token_id in [Token.FLOAT]:
            self.eat(Token.FLOAT)  # Consume 'float'
            id_name = self.to_match.value
            new_name = self.nng.mk_new_name()
            data = SymbolTableData(IDType.VAR, Type.FLOAT, new_name)
            self.symbol_table.ht_stack[-1][id_name] = data
            self.eat(Token.ID)  # Consume identifier
            self.eat(Token.SEMI)  # Consume semicolon
            return []

        # Unexpected token
        raise ParserException(self.scanner.get_lineno(), self.to_match, [Token.INT, Token.FLOAT])


    def parse_assignment_statement(self) -> List[str]:
        code = self.parse_assignment_statement_base()
        self.eat(Token.SEMI)
        return code

    def parse_assignment_statement_base(self) -> List[str]:
        id_name = self.to_match.value
        id_data = self.symbol_table.lookup(id_name)
        if id_data == None:
            raise SymbolTableException(self.scanner.get_lineno(), id_name)
        self.eat(Token.ID)  # LHS variable
        self.eat(Token.ASSIGN)  # '='

        expr = self.parse_expr()  # Parse expression on RHS
        type_inference(expr)
        map_vregs(expr, self.vra)
        ast_lines = expr.linearize()  # Get 3AC
        data_name = id_data.get_new_name()
        data_type = id_data.get_data_type()
        expr_type = expr.node_type

        # Handle IO variable assignments with conversions
        if id_data.get_id_type() == IDType.IO:
            if data_type == Type.INT and expr_type == Type.FLOAT:
                temp_vr = self.vra.mk_new_vr()
                ast_lines.append(f"{temp_vr} = vr_float2int({expr.vr});")
                ast_lines.append(f"{data_name} = vr2int({temp_vr});")
            elif data_type == Type.FLOAT and expr_type == Type.INT:
                temp_vr = self.vra.mk_new_vr()
                ast_lines.append(f"{temp_vr} = vr_int2float({expr.vr});")
                ast_lines.append(f"{data_name} = vr2float({temp_vr});")
            else:
                if data_type == Type.INT:
                    ast_lines.append(f"{data_name} = vr2int({expr.vr});")
                else:
                    ast_lines.append(f"{data_name} = vr2float({expr.vr});")
        else:
            # Handle regular variables with conversion if needed
            if data_type == Type.INT and expr_type == Type.FLOAT:
                temp_vr = self.vra.mk_new_vr()
                ast_lines.append(f"{temp_vr} = vr_float2int({expr.vr});")
                ast_lines.append(f"{data_name} = {temp_vr};")
            elif data_type == Type.FLOAT and expr_type == Type.INT:
                temp_vr = self.vra.mk_new_vr()
                ast_lines.append(f"{temp_vr} = vr_int2float({expr.vr});")
                ast_lines.append(f"{data_name} = {temp_vr};")
            else:
                ast_lines.append(f"{data_name} = {expr.vr};")
        return ast_lines

    def parse_if_else_statement(self) -> List[str]:
        self.eat(Token.IF)
        self.eat(Token.LPAR)
        expr = self.parse_expr()
        type_inference(expr)
        map_vregs(expr, self.vra)
        ast_lines = expr.linearize()
        self.eat(Token.RPAR)

        label_begin = self.nlg.mk_new_label()
        label_end = self.nlg.mk_new_label()

        new_vr = self.vra.mk_new_vr()
        ast_lines.append(f"{new_vr} = int2vr(0);")
        ast_lines.append(f"beq({new_vr}, {expr.vr}, {label_begin});")  # Branch if false

        true_lines = self.parse_statement()
        if not isinstance(true_lines, list): true_lines = []

        self.eat(Token.ELSE)

        gap = [f"branch({label_end});\n{label_begin}:"]
        false_lines = self.parse_statement()
        if not isinstance(false_lines, list): false_lines = []

        return ast_lines + true_lines + gap + false_lines + [f"{label_end}:"]


    def parse_block_statement(self) -> List[str]:
        self.eat(Token.LBRACE)
        self.symbol_table.push_scope()
        code = self.parse_statement_list()
        self.symbol_table.pop_scope()
        self.eat(Token.RBRACE)
        return code

    def parse_for_statement(self) -> List[str]:
        self.eat(Token.FOR)
        self.eat(Token.LPAR)

        ast_init_lines = self.parse_assignment_statement()  # Init
        expr = self.parse_expr()  # Condition
        type_inference(expr)
        map_vregs(expr, self.vra)
        ast_lines = expr.linearize()

        self.eat(Token.SEMI)
        ast_updated_lines = self.parse_assignment_statement_base()  # Update
        self.eat(Token.RPAR)

        label_begin = self.nlg.mk_new_label()
        label_end = self.nlg.mk_new_label()

        ast_body_lines = self.parse_statement()  # Loop body

        new_vr = self.vra.mk_new_vr()
        expr_gap = ast_lines + [f"{new_vr} = int2vr(0);\nbeq({new_vr}, {expr.vr}, {label_end});"]

        return (
            ast_init_lines +
            [f"{label_begin}:"] +
            expr_gap +
            ast_body_lines +
            ast_updated_lines +
            [f"branch({label_begin});\n{label_end}:"]
        )

    def parse_expr(self) -> ASTNode:
        left = self.parse_comp()
        return self.parse_expr2(left)

    def parse_expr2(self, left: ASTNode) -> ASTNode:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.EQ]:
            self.eat(Token.EQ)
            right = self.parse_comp()
            new_node = ASTEqNode(left, right)
            return self.parse_expr2(new_node)
        if token_id in [Token.SEMI, Token.RPAR]:
            return left
        raise ParserException(self.scanner.get_lineno(), self.to_match,
                              [Token.EQ, Token.SEMI, Token.RPAR])

    def parse_comp(self) -> ASTNode:
        left = self.parse_factor()
        return self.parse_comp2(left)

    def parse_comp2(self, left: ASTNode) -> ASTNode:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.LT]:
            self.eat(Token.LT)
            right = self.parse_factor()
            new_node = ASTLtNode(left, right)
            return self.parse_comp2(new_node)
        if token_id in [Token.SEMI, Token.RPAR, Token.EQ]:
            return left
        raise ParserException(self.scanner.get_lineno(), self.to_match,
                              [Token.EQ, Token.SEMI, Token.RPAR, Token.LT])

    def parse_factor(self) -> ASTNode:
        left = self.parse_term()
        return self.parse_factor2(left)

    def parse_factor2(self, left: ASTNode) -> ASTNode:
        token_id = self.get_token_id(self.to_match)
        if token_id == Token.PLUS:
            self.eat(Token.PLUS)
            right = self.parse_term()
            new_node = ASTPlusNode(left, right)
            return self.parse_factor2(new_node)
        if token_id == Token.MINUS:
            self.eat(Token.MINUS)
            right = self.parse_term()
            new_node = ASTMinusNode(left, right)
            return self.parse_factor2(new_node)
        if token_id in [Token.EQ, Token.SEMI, Token.RPAR, Token.LT]:
            return left
        raise ParserException(self.scanner.get_lineno(), self.to_match,
                              [Token.EQ, Token.SEMI, Token.RPAR, Token.LT, Token.PLUS, Token.MINUS])

    def parse_term(self) -> ASTNode:
        left = self.parse_unit()
        return self.parse_term2(left)

    def parse_term2(self, left: ASTNode) -> ASTNode:
        token_id = self.get_token_id(self.to_match)
        if token_id == Token.DIV:
            self.eat(Token.DIV)
            right = self.parse_unit()
            new_node = ASTDivNode(left, right)
            return self.parse_term2(new_node)
        if token_id == Token.MUL:
            self.eat(Token.MUL)
            right = self.parse_unit()
            new_node = ASTMultNode(left, right)
            return self.parse_term2(new_node)
        if token_id in [Token.EQ, Token.SEMI, Token.RPAR, Token.LT, Token.PLUS, Token.MINUS]:
            return left
        raise ParserException(self.scanner.get_lineno(), self.to_match,
                              [Token.EQ, Token.SEMI, Token.RPAR, Token.LT, Token.PLUS, Token.MINUS, Token.MUL, Token.DIV])

    def parse_unit(self) -> ASTNode:
        token_id = self.get_token_id(self.to_match)
        if token_id == Token.NUM:
            num_value = self.to_match.value
            self.eat(Token.NUM)
            return ASTNumNode(num_value)
        if token_id == Token.ID:
            id_name = self.to_match.value
            id_data = self.symbol_table.lookup(id_name)
            if id_data is None:
                raise SymbolTableException(self.scanner.get_lineno(), id_name)
            self.eat(Token.ID)
            if id_data.get_id_type() == IDType.IO:
                return ASTIOIDNode(id_data.get_new_name(), id_data.get_data_type())
            else:
                return ASTVarIDNode(id_data.get_new_name(), id_data.get_data_type())
        if token_id == Token.LPAR:
            self.eat(Token.LPAR)
            expr = self.parse_expr()
            self.eat(Token.RPAR)
            return expr
        raise ParserException(self.scanner.get_lineno(), self.to_match,
                              [Token.NUM, Token.ID, Token.LPAR])

def is_leaf_node(node) -> bool:
    return issubclass(type(node),ASTLeafNode)

def type_inference(node: ASTNode) -> Type:
    if is_leaf_node(node):
        return node.node_type

    # Handle unary ops
    if isinstance(node, ASTIntToFloatNode):
        node.node_type = Type.FLOAT
        return Type.FLOAT
    elif isinstance(node, ASTFloatToIntNode):
        node.node_type = Type.INT
        return Type.INT

    # Handle binary ops
    elif isinstance(node, ASTBinOpNode):
        left_type = type_inference(node.l_child)
        right_type = type_inference(node.r_child)

        if isinstance(node, ASTEqNode) or isinstance(node, ASTLtNode):
            node.node_type = Type.INT
            if left_type == Type.FLOAT or right_type == Type.FLOAT:
                if left_type != Type.FLOAT and left_type == Type.INT:
                    node.l_child = ASTIntToFloatNode(node.l_child)
                if right_type != Type.FLOAT and right_type == Type.INT:
                    node.r_child = ASTIntToFloatNode(node.r_child)
            return Type.INT

        if left_type == Type.FLOAT or right_type == Type.FLOAT:
            node.node_type = Type.FLOAT
            if left_type != Type.FLOAT and left_type == Type.INT:
                node.l_child = ASTIntToFloatNode(node.l_child)
            if right_type != Type.FLOAT and right_type == Type.INT:
                node.r_child = ASTIntToFloatNode(node.r_child)
            return Type.FLOAT

        node.node_type = Type.INT
        return Type.INT

# Assigns virtual registers to AST nodes
def map_vregs(node: ASTNode, vr_allocator: VRAllocator) -> None:
    if isinstance(node, ASTLeafNode):
        if node.value.startswith("vr"):
            node.vr = node.value
        else:
            node.vr = vr_allocator.mk_new_vr()
        return
    if isinstance(node, ASTBinOpNode):
        map_vregs(node.l_child, vr_allocator)
        map_vregs(node.r_child, vr_allocator)
        node.vr = vr_allocator.mk_new_vr()
        return
    elif isinstance(node, ASTUnOpNode):
        map_vregs(node.child, vr_allocator)
        node.vr = vr_allocator.mk_new_vr()
        return
    elif isinstance(node, ASTVarIDNode):
        node.vr = vr_allocator.mk_new_vr()
        return
