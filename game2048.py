import pygame 
import random
import math
from time import time
pygame.init()

# static methods

def timer(f):
    
    def wrap(*args, **kwargs):
        time_start = time()
        res = f(*args,**kwargs)
        time_end = time()
        print(f"Took {time_end-time_start} to run {str(f)}")
        return res
    return wrap


FPS = 60
WIDTH, HEIGHT = 800, 800

ROWS = 4
COLS = 4

RECT_HEIGHT = HEIGHT//ROWS
RECT_WIDTH = WIDTH//ROWS

OUTLINE_COLOR = (187,173,160)
OUTLINE_THICKNESS = 10

BACKGROUND_COLOR = (205,192,180)
FONT_COLOR = (119,110,101)


FONT = pygame.font.SysFont("comicsans",60,bold=True)
SCORE_FONT = pygame.font.SysFont("comicsans",32,bold=True)
MOVE_VEL = 20 #px per sec

WINDOW = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("2048")




class Game:
    
    def __init__(self):
        self.score=0
        self.high_score = Game.get_high_score()
        
    def update_score(self,val):
        self.score+=val
        if self.score > self.high_score:
            Game.set_high_score(self.score)
            self.high_score = self.score
          
        
    @staticmethod
    def get_high_score():
        f = open("score.txt","r")
        return int(f.read())
    
    @staticmethod
    def set_high_score(val):
        f = open("score.txt","w")
        f.write(str(val))


class Tile:
    COLORS = [

        (237,229,218),
        (238,225,201),
        (243,178,122),
        (246,150,101),
        (247,124,95),
        (247,95,59),
        (237,208,115),
        (237,204,99),
        (236,202,80),
        (236,111,80),
        (236,111,255)

        

    ]
    
    def __init__(self,value,row,col):
        self.value = value
        self.row = row
        self.col = col
        self.x = col * RECT_HEIGHT
        self.y = row * RECT_WIDTH
        
    def get_color(self):
        color_index = int(math.log2(self.value)) -1
        color = self.COLORS[color_index]
        return color
        
    def draw(self,window):
        color = self.get_color()
        pygame.draw.rect(window,color,(self.x,self.y,RECT_WIDTH,RECT_HEIGHT))
        
        text = FONT.render(str(self.value),1,FONT_COLOR)
        
        window.blit(
            text,
            (self.x + (RECT_WIDTH/2) - text.get_width()/2,
             self.y + (RECT_HEIGHT/2) - text.get_height()/2),
        )
        
        
    def set_pos(self,ceil=False):
        
        if ceil:
            self.row = math.ceil(self.y/RECT_HEIGHT)
            self.col = math.ceil(self.x/RECT_WIDTH)
        else: 
            self.row = math.floor(self.y/RECT_HEIGHT)
            self.col = math.floor(self.x/RECT_WIDTH)
            
        
    
    
    def move(self,delta):
        self.x += delta[0]
        self.y += delta[1]


        
def draw_grid(window):
    for row in range(1,ROWS):
        y = row * RECT_HEIGHT
        pygame.draw.line(window,OUTLINE_COLOR,(0,y),(WIDTH,y),OUTLINE_THICKNESS)
        
    for col in range(1,COLS):
        x = col * RECT_WIDTH
        pygame.draw.line(window,OUTLINE_COLOR,(x,0),(x,HEIGHT),OUTLINE_THICKNESS)

    pygame.draw.rect(window,OUTLINE_COLOR,(0,0,WIDTH,HEIGHT),OUTLINE_THICKNESS)

def draw_score(window,game):
    score_text = SCORE_FONT.render(f"Score: {game.score} High Score: {game.high_score}", True, FONT_COLOR)
    
    
    window.blit(score_text,(10,10))

def draw(window,tiles,game):
    window.fill(BACKGROUND_COLOR)
    
    for tile in tiles.values(): 
        tile.draw(window) 
    
    draw_grid(window)
    draw_score(window,game)
 
    pygame.display.update()
    
    

def get_random_pos(tiles):
    row = None
    col = None
    while True:
        row = random.randrange(0,ROWS)
        col = random.randrange(0,COLS)
        
        if f"{row}{col}" not in tiles:
            break
    return row,col


def generate_tiles():
    tiles = {}
    
    for _ in range(2):
        row,col = get_random_pos(tiles)
        tiles[f"{row}{col}"] = Tile(2,row,col)
        
    return tiles

def move_tiles(window,tiles,clock,direction,game):
    updated = True
    blocks = set()
    
    if direction == "left":
        sort_func = lambda x: x.col
        reverse = False
        delta = (-MOVE_VEL, 0)
        boundary_check = lambda tile: tile.col == 0
        get_next_tile = lambda tile: tiles.get(f"{tile.row}{tile.col - 1}")
        merge_check = lambda tile, next_tile: tile.x > next_tile.x + MOVE_VEL
        move_check = (
            lambda tile, next_tile: tile.x > next_tile.x + RECT_WIDTH + MOVE_VEL
        )
        ceil = True
    elif direction == "right":
        sort_func = lambda x: x.col
        reverse = True
        delta = (MOVE_VEL, 0)
        boundary_check = lambda tile: tile.col == COLS - 1
        get_next_tile = lambda tile: tiles.get(f"{tile.row}{tile.col + 1}")
        merge_check = lambda tile, next_tile: tile.x < next_tile.x - MOVE_VEL
        move_check = (
            lambda tile, next_tile: tile.x + RECT_WIDTH + MOVE_VEL < next_tile.x
        )
        ceil = False
    elif direction == "up":
        sort_func = lambda x: x.row
        reverse = False
        delta = (0, -MOVE_VEL)
        boundary_check = lambda tile: tile.row == 0
        get_next_tile = lambda tile: tiles.get(f"{tile.row - 1}{tile.col}")
        merge_check = lambda tile, next_tile: tile.y > next_tile.y + MOVE_VEL
        move_check = (
            lambda tile, next_tile: tile.y > next_tile.y + RECT_HEIGHT + MOVE_VEL
        )
        ceil = True
    elif direction == "down":
        sort_func = lambda x: x.row
        reverse = True
        delta = (0, MOVE_VEL)
        boundary_check = lambda tile: tile.row == ROWS - 1
        get_next_tile = lambda tile: tiles.get(f"{tile.row + 1}{tile.col}")
        merge_check = lambda tile, next_tile: tile.y < next_tile.y - MOVE_VEL
        move_check = (
            lambda tile, next_tile: tile.y + RECT_HEIGHT + MOVE_VEL < next_tile.y
        )
        ceil = False
    score_up=0
    win_detected = False
    while updated:
        clock.tick(FPS)
        updated = False
        sorted_tiles = sorted(tiles.values(), key=sort_func, reverse=reverse)
        for i, tile in enumerate(sorted_tiles):
            if boundary_check(tile):
                continue

            next_tile = get_next_tile(tile)
            if not next_tile:
                tile.move(delta)
            elif (
                tile.value == next_tile.value
                and tile not in blocks
                and next_tile not in blocks
            ):
                if merge_check(tile, next_tile):
                    tile.move(delta)
                else:
                    next_tile.value *= 2
                    score_up += next_tile.value
                    if next_tile.value == 2048:
                        win_detected = True
                    sorted_tiles.pop(i)
                    blocks.add(next_tile)
            elif move_check(tile, next_tile):
                tile.move(delta)
            else:
                continue

            tile.set_pos(ceil)
            updated = True

        update_tiles(window, tiles, sorted_tiles,game)
    if win_detected:
        print("You win!")
    return end_move(tiles), score_up
    



# @timer    
def check_loss(tiles):
    # print("checking length")
    if len(tiles) < 16:
        return False
    # print("checking loss")
    board = [[None for _ in range(COLS)] for _ in range(ROWS)]
    # print("created board")
    for _,tile in tiles.items():
        board[tile.row][tile.col] = tile.value
    # print("filled board")
        
    for row in  range(ROWS):
        for col in range(COLS):
            val = board[row][col]
            if col < COLS-1 and board[row][col+1] == val:
                # print("we good")
                return False
            if row < ROWS-1 and board[row+1][col] == val:
                # print("we good")
                return False
    # print("we good")
    return True
def end_move(tiles):
    # print("ending move")
    if check_loss(tiles):
        print("lost")
        return "lost"
    if len(tiles) < 16:
        row,col = get_random_pos(tiles)
        tiles[f"{row}{col}"] = Tile(random.choice([2,4]),row,col)
    return "continue"
        
        
def update_tiles(window,tiles,sorted_tiles,game):
    tiles.clear()
    for tile in sorted_tiles:
        tiles[f"{tile.row}{tile.col}"] = tile
    draw(window,tiles,game)        
            
         
def main(window):
    clock = pygame.time.Clock()
    run = True
    score=0
    tiles = generate_tiles()
    game = Game()
    while run:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    move_result = move_tiles(window,tiles,clock,"left",game)                    
                if event.key == pygame.K_RIGHT:
                    move_result = move_tiles(window,tiles,clock,"right",game)
                if event.key == pygame.K_DOWN:
                    move_result = move_tiles(window,tiles,clock,"down",game)
                if event.key == pygame.K_UP:
                    move_result = move_tiles(window,tiles,clock,"up",game)
                
                game.update_score(move_result[1])
                
                
                if (move_result[0] == "lost") or ( move_result[1] == 2048):
                    print(f"Game Over\nScore: {game.score}" )
                    run = False
                
        
        draw(WINDOW,tiles,game)
    pygame.quit()
    pass


if __name__ == "__main__":
    main(WINDOW)
