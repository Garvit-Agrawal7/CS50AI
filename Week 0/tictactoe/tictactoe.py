import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = 0
    o_count = 0

    # Return whose turn is first by checking who's made more moves
    for i in board:
        x_count += i.count(X)
        o_count += i.count(O)
    if x_count > o_count:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] is None:
                possible_actions.add((i, j))
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    test_board = copy.deepcopy(board)
    if (action[0], action[1]) not in actions(board):
        raise Exception("Not a legal move")
    else:
        test_board[action[0]][action[1]] = player(board)
        return test_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for y in range(len(board)):
        for x in range(len(board[y])):
            if board[y].count(X) == 3 or board[y].count(O) == 3:
                return board[y][x]
    if board[0][0] == board[1][0] == board[2][0]:
        return board[0][0]
    elif board[0][1] == board[1][1] == board[2][1]:
        return board[0][1]
    elif board[0][2] == board[1][2] == board[2][2]:
        return board[0][2]
    elif board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    elif board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]
    else:
        return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    elif winner(board) is None and not any(None in y for y in board):
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) is X:
        return 1
    elif winner(board) is O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    else:
        if player(board) is X:
            return max_value(board)[1]
        else:
            return min_value(board)[1]


def max_value(board):
    if terminal(board):
        return utility(board), None
    v = float('-inf')
    optimal_action = None
    for action in actions(board):
        new_value = min_value(result(board, action))
        value = new_value[0]
        if value > v:
            v = value
            optimal_action = action
        if v == 1:
            return v, optimal_action
    return v, optimal_action


def min_value(board):
    if terminal(board):
        return utility(board), None
    v = float('inf')
    optimal_actions = None
    for action in actions(board):
        new_value = max_value(result(board, action))
        value = new_value[0]
        if value < v:
            v = value
            optimal_actions = action
            if v == -1:
                return v, optimal_actions
    return v, optimal_actions
