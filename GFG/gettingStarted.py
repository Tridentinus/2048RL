import pygame 
  
pygame.init() 

# Design Colors
color = (255, 255, 255)
rectangle_color = (0, 255, 0)
position = (0,0)
  
# CREATING CANVAS 
canvas = pygame.display.set_mode((500, 500)) 

# TITLE OF CANVAS 
pygame.display.set_caption("My Board") 

image = pygame.image.load('ss.png')
exit = False
  
while not exit: 
    canvas.fill(color)
    for event in pygame.event.get(): 
        if event.type == pygame.QUIT: 
            exit = True
    pygame.draw.rect(canvas, rectangle_color, pygame.Rect(30, 30, 60, 60))
    pygame.display.update() 
