from PIL import Image
import win32api, win32con
import time
import cv2
from mss.windows import MSS as mss
import numpy as np


LEFT, RIGHT, TOP, BOTTOM = 0, 1, 2, 3
SNAPSHOT_AREA = {'top': 162, 'left': 608, 'width': 794, 'height': 700} ## Real
## SNAPSHOT_AREA = {'top': 109, 'left': 132, 'width': 794, 'height': 700} ## Testing

COLORS_BGR = {"Red": (1, 2, 246), "RS": (67, 70, 243), "RS2": (97, 101, 247), "RW": (36, 35, 253),
          "Green" : (2, 181, 54), "GS": (72, 237, 109), "GS2": (126, 244, 163), "GS3": (107, 242, 143), "GW": (47, 229, 91),
          "Blue": (255, 154, 44), "BS": (244, 205,  85), "BW": (255, 196,  36),
          "Yellow": (12, 225, 252), "YS": (76, 221, 250),
          "Purple": (255,  35, 195), "PS": (245, 107, 215),
          "Orange": (35, 155, 255), "OS": (119, 203, 249),
          "Colored": (45, 69, 112)
         }

# to keep track of possible swaps and their scores
POSSIBLE_SWAPS = {}

# minimum and maximum number of candies that can be matched
MIN_MATCH = 3
MAX_MATCH = 5

# horiz / vert distances between cells in pixels
HRANGE = 89
VRANGE = 79

# middle of top left cell in pixels
TOP_LEFT = (40, 35)

COLS = 9
ROWS = 9

# The values in screen coordinates
# used by win32 functions
VRANGE_SC = 63
HRANGE_SC = 71
TOP_LEFT_SC = (517, 150) ## Real
## TOP_LEFT_SC = (138,110) ## Testing

sct = mss()

# Finds mean BGR color value of [bgr_img]
def meanBgr(bgr_img):
    color_mean = np.average(bgr_img, axis = 0)
    color_mean = np.average(color_mean, axis = 0)
    color_mean = np.uint8([[color_mean]])
    return color_mean[0][0]
#
# get mean BGR color value of candy at position r,c in gameboard
# returns as (b,g,r) set
def getCandyColor(r,c,img):
    x = (TOP_LEFT[0] + (c-1) * HRANGE) - 10
    y = (TOP_LEFT[1] + (r-1) * VRANGE) - 10
    bgr_img = img[y:y+20, x:x+20]
    mean_px = meanBgr(bgr_img)
    return mean_px
#
# categorize [color] into one from COLORS_BGR using manhattan distances
# if distance is above a threshold, returns '?'
def categorizeColor(bgr_tuple):
    manhattan = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2])
    distances = {k: manhattan(v, bgr_tuple) for k, v in COLORS_BGR.iteritems()}
    color = min(distances, key=distances.get)
    threshold = 40
    if not distances[color] > threshold:
        return color[0]
    return '?'
#
# From r,c cell of gameboard, check [direction] for pattern with [sequence-1] # of same candies and 1 unique candy
# If such pattern is found, call findSwap on unique candy position
def findPattern(board, r, c, sequence, direction):
    matches = {}
    if (direction == LEFT):
        i = c-(sequence-1)
        while (i <= c):
            seen = [x[0] for x in matches.keys()]
            if not board[r][i] in seen:
                matches[(board[r][i],i)] = 1
            else:
                matches[matches.keys()[seen.index(board[r][i])]] += 1
            i += 1
        if (sequence-1 in matches.values() and 1 in matches.values() and len(matches.keys()) == 2):
            target = matches.keys()[matches.values().index(sequence-1)][0]
            C = matches.keys()[matches.values().index(1)][1]
            if (C == c):
                restrict = (1,0,0,0)
            elif (C == c-(sequence-1)):
                restrict = (0,1,0,0)
            else:
                restrict = (1,1,0,0)
            findSwap(board, r, C, target, restrict)
    if (direction == RIGHT):
        i = c
        while (i <= c+(sequence-1)):
            seen = [x[0] for x in matches.keys()]
            if not board[r][i] in seen:
                matches[(board[r][i],i)] = 1
            else:
                matches[matches.keys()[seen.index(board[r][i])]] += 1
            i += 1
        if (sequence-1 in matches.values() and 1 in matches.values() and len(matches.keys()) == 2):
            target = matches.keys()[matches.values().index(sequence-1)][0]
            C = matches.keys()[matches.values().index(1)][1]
            if (C == c):
                restrict = (0,1,0,0)
            elif (C == c+(sequence-1)):
                restrict = (1,0,0,0)
            else:
                restrict = (1,1,0,0)
            findSwap(board, r, C, target, restrict)
    if (direction == TOP):
        i = r
        while (i >= r-(sequence-1)):
            seen = [x[0] for x in matches.keys()]
            if not board[i][c] in seen:
                matches[(board[i][c],i)] = 1
            else:
                matches[matches.keys()[seen.index(board[i][c])]] += 1
            i -= 1
        if (sequence-1 in matches.values() and 1 in matches.values() and len(matches.keys()) == 2):
            target = matches.keys()[matches.values().index(sequence-1)][0]
            R = matches.keys()[matches.values().index(1)][1]
            if (R == r):
                restrict = (0,0,1,0)
            elif (R == r-(sequence-1)):
                restrict = (0,0,0,1)
            else:
                restrict = (0,0,1,1)
            findSwap(board, R, c, target, restrict)
    if (direction == BOTTOM):
        i = r+(sequence-1)
        while (i >= r):
            seen = [x[0] for x in matches.keys()]
            if not board[i][c] in seen:
                matches[(board[i][c],i)] = 1
            else:
                matches[matches.keys()[seen.index(board[i][c])]] += 1
            i -= 1
        if (sequence-1 in matches.values() and 1 in matches.values() and len(matches.keys()) == 2):
            target = matches.keys()[matches.values().index(sequence-1)][0]
            R = matches.keys()[matches.values().index(1)][1]
            if (R == r):
                restrict = (0,0,0,1)
            elif (R == r+(sequence-1)):
                restrict = (0,0,1,0)
            else:
                restrict = (0,0,1,1)
            findSwap(board, R, c, target, restrict)
#
# From r,c cell of gameboard, check one step on all four sides for candy match of [target]
# Restrict is a tuple specifying if a direction should not be checked
# If target candy match found, add the possible swap to the global dict 
def findSwap(board, r, c, target, restrict):
    if (board[r][c] != '?'):
        if (r > 0 and not restrict[TOP] and board[r-1][c] == target):
            POSSIBLE_SWAPS[(r,c,r-1,c)] = 0
        if (r < ROWS-1 and not restrict[BOTTOM] and board[r+1][c] == target):
            POSSIBLE_SWAPS[(r,c,r+1,c)] = 0
        if (c > 0 and not restrict[LEFT] and board[r][c-1] == target):
            POSSIBLE_SWAPS[(r,c,r,c-1)] = 0
        if (c < COLS-1 and not restrict[RIGHT] and board[r][c+1] == target):
            POSSIBLE_SWAPS[(r,c,r,c+1)] = 0
#
# For each cell (from bottom left) in gameboard, call findPattern (with sequence parameter) four times for each direction
def findPatterns(gameboard, sequence):
    R = ROWS-1
    C = 0
    while (R >= 0):
        while (C < COLS):
            if (gameboard[R][C] != '?'):
                if (R >= (sequence-1) and columnsStable(gameboard, [C-1,C,C+1])):
                    findPattern(gameboard, R, C, sequence, TOP)
                if (R <= ROWS-sequence and columnsStable(gameboard, [C-1,C,C+1])):
                    findPattern(gameboard, R, C, sequence, BOTTOM)
                if (C >= (sequence-1) and columnsStable(gameboard, [c for c in range(C,C-sequence,-1)])):
                    findPattern(gameboard, R, C, sequence, LEFT)
                if (C <= COLS-sequence and columnsStable(gameboard, [c for c in range(C,C+sequence)])):
                    findPattern(gameboard, R, C, sequence, RIGHT)
            C += 1
        R -= 1
        C = 0
#
# Checks if all candies in the specified columns are in place (being detected)
# ie. no candies are still falling down / missing
def columnsStable(board, cols):
    return True # nop the func for now
    cells = [x for row in board for x in row if row.index(x) in cols]
    if not '?' in cells:
        return True
    return False
#
# Simulate mouse drag to swap (r1,c1) with (r2,c2) on screen gameboard
def makeSwap(r1, c1, r2, c2):
    x1 = TOP_LEFT_SC[0] + (HRANGE_SC * c1)
    x2 = TOP_LEFT_SC[0] + (HRANGE_SC * c2)
    y1 = TOP_LEFT_SC[1] + (VRANGE_SC * r1)
    y2 = TOP_LEFT_SC[1] + (VRANGE_SC * r2)
    win32api.SetCursorPos((x1,y1))
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
    time.sleep(0.10)
    win32api.SetCursorPos((x2,y2))
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)
#
# Returns a copy of gameboard with cell at pos (r1, c1) swapped with (r2, c2)
def pretendSwap(board, r1, c1, r2, c2):
    newBoard = [row[:] for row in gameboard]
    first = newBoard[r1][c1]
    newBoard[r1][c1] = newBoard[r2][c2]
    newBoard[r2][c2] = first
    return newBoard

# From r,c pos in gameboard, check [direction] for any number of candies with matching color
# returns number of candies with matching color and 0 if there were none
def findMatch(board, r, c, direction):
    matchLength = 0
    if (direction == LEFT):
        while (c > 0):
            cur = board[r][c]
            nxt = board[r][c-1]
            if (cur == nxt):
                matchLength += 1
                c -=1
            else:
                break
    if (direction == RIGHT):
        while (c < COLS-1):
            cur = board[r][c]
            nxt = board[r][c+1]
            if (cur == nxt):
                matchLength += 1
                c +=1
            else:
                break
    if (direction == TOP):
        while (r > 0):
            cur = board[r][c]
            nxt = board[r-1][c]
            if (cur == nxt):
                matchLength += 1
                r -=1
            else:
                break
    if (direction == BOTTOM):
        while (r < ROWS-1):
            cur = board[r][c]
            nxt = board[r+1][c]
            if (cur == nxt):
                matchLength += 1
                r +=1
            else:
                break
    return matchLength + 1
#
# For each cell in gameboard, call findMatch four times for each direction
# Returns set of tuples containing all matched candies positions
def findMatches(board, minMatch):
    matched = []
    exclude = ('X', '?')
    for r in range(ROWS):
        for c in range(COLS):
            if (c >= minMatch-1 and (not board[r][c] in exclude)):
                match = findMatch(board, r, c, LEFT)
                if (match >= minMatch):
                    matched += [(r, x) for x in range(c, c-match, -1)]

            if (c <= COLS-minMatch and (not board[r][c] in exclude)):
                match = findMatch(board, r, c, RIGHT)
                if (match >= minMatch):
                    matched += [(r, x) for x in range(c, c+match)]

            if (r >= minMatch-1 and (not board[r][c] in exclude)):
                match = findMatch(board, r, c, TOP)
                if (match >= minMatch):
                    matched += [(x, c) for x in range(r, r-match, -1)]

            if (r <= ROWS-minMatch and (not board[r][c] in exclude)):
                match = findMatch(board, r, c, BOTTOM)
                if (match >= minMatch):
                    matched += [(x, c) for x in range(r, r+match)]
    return set(matched)

# removes all [matched] candies from board
# adjusts position of other candies accordingly
def simulateBoard(board, matched):
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in matched:
                board[r][c] = 'X'

    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == 'X' and r > 0:
                R = r
                while (R > 0):
                    board[R][c] = board[R-1][c]
                    board[R-1][c] = 'X'
                    R -= 1

# calculate score of swapping two candies on board taking into account all cominations of matches
# append result to [POSSIBLE_SWAPS] global dict with (r1,c1,r2,c2) as key and score as value
def findSwapScore(board, r1, c1, r2, c2):
    newBoard = pretendSwap(board, r1, c1, r2, c2)
    totalScore = 0
    while True:
        matches = findMatches(newBoard, MIN_MATCH)
        score = len(matches)
        if (score == 0):
            break
        totalScore += score
        simulateBoard(newBoard, matches)
    POSSIBLE_SWAPS[(r1,c1,r2,c2)] = totalScore
#
# Finds the score of all possible swaps and carries out the one with the higest score
# if multiple swaps give the same highest score, carry out the one thats furthest down
def makeBestMove(gameboard):
    maximum = 0
    moves = []
    for (r1, c1, r2, c2) in POSSIBLE_SWAPS.keys():
        findSwapScore(gameboard, r1, c1, r2, c2)
        score = POSSIBLE_SWAPS[(r1, c1, r2, c2)]
        if (min(r1,r2) <= 3): # penalize for making swaps on uppper portion of board
            score -= (4-min(r1,r2))
        if score > maximum: # if new max score found, reset list of possible moves
            moves = []
        if score >= maximum:
            maximum = score
            moves.append((r1,c1,r2,c2))
    if (len(moves) > 0):
        moveLocs = [min(y1,y2) for (y1,x1,y2,x2) in moves]
        best = moves[moveLocs.index(max(moveLocs))]
        R1, C1, R2, C2 = best
        makeSwap(R1, C1, R2, C2)
#
# print detected candies from screen gameboard
def debug(gameboard, image):
    unknown = []
    for y in range(ROWS):
        for x in range(COLS):
            val = gameboard[y][x]
            if val == '?':
                color = getCandyColor(y+1, x+1, image)
                unknown.append([color, y+1, x+1])
            print gameboard[y][x],
        print '\n'
    for entry in unknown:
            print entry
    print '-' * 18 + '\n'
#
# show section of screen that was cropped
def debugImg(img):
    while True:
        cv2.imshow('test', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
#
# Read pixel colors from game screen and initialize gameboard
def initializeBoard(gameboard):
    sct.get_pixels(SNAPSHOT_AREA)
    image = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    for r in range(ROWS):
        for c in range(COLS):
            colorVal = getCandyColor(r+1, c+1, image)
            colorName = categorizeColor(colorVal)
            gameboard[r][c] = colorName
    if win32api.GetAsyncKeyState(ord('D')) < 0:
        debug(gameboard, image)
#
# Entry point of program
#
print "Running ..."
while not win32api.GetAsyncKeyState(ord('Q')) < 0:
    POSSIBLE_SWAPS = {}
    gameboard = [[0 for c in range(COLS)] for r in range(ROWS)]
    initializeBoard(gameboard)
    
    #if not '?' in [x for row in gameboard for x in row]:
    for i in range(MAX_MATCH, MIN_MATCH-1, -1):
        findPatterns(gameboard, i)
        if len(POSSIBLE_SWAPS.keys()) > 0:
            makeBestMove(gameboard)
            break
    