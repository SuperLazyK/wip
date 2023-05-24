from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import pygame.locals as pl
import numpy as np
import sys

#-------------------------
# Draw
#-------------------------

def arr2txt(a, title=""):
    a2 = a.reshape(-1)
    return " ".join([f"{title}[{i}]: {a2[i]:.03f}" for i in range(a2.shape[0])])

SCREEN_SIZE=(1300, 500)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def default_event_handler(key, shifted):
    if key == 'q':
        sys.exit()

def s2color(s):
    if s == "white":
        return (255, 255, 255)
    elif s == "red":
        return (170, 0, 0)
    elif s == "green":
        return (0, 170, 0)
    elif s == "blue":
        return (0, 0, 170)
    else:
        return (0, 0, 0)

class Viewer():
    def __init__(self, scale=1, screen_size=SCREEN_SIZE, offset=[0, 0]):
        pygame.init()
        pygame.event.clear()
        self.scale = scale
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(self.screen_size)
        self.offset = offset
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Calibri', 25, True, False)

    def text(self, ss, color=None):
        for i, s in enumerate(ss):
            text = self.font.render(s, True, s2color(color))
            self.screen.blit(text, [100, i*50+100])

    def conv_pos(self, p):
        ret = self.scale * np.array([p[0], -p[1]]) + np.array(self.screen_size)/2 +  np.array([self.screen_size[0] * self.offset[0], self.screen_size[1] * self.offset[1]])
        return ret

    def clear(self):
        self.screen.fill(WHITE)

    def handle_event(self, handler):
        for event in pygame.event.get():
            if event.type == pl.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pl.KEYDOWN:
                keyname = pygame.key.name(event.key)
                mods = pygame.key.get_mods()
                handler(keyname, mods & pl.KMOD_LSHIFT)

    def draw_horizon(self, y):
        pygame.draw.line(self.screen, BLACK, self.conv_pos((-1000,y)), self.conv_pos((1000,y)),  width=int(2))

    def draw(self, cmds):
        for cmd in cmds:
            if cmd["type"] == "lineseg":
                pygame.draw.line(self.screen, s2color(cmd.get("color")), self.conv_pos(cmd["start"]), self.conv_pos(cmd["end"]),  width=int(1))
            elif cmd["type"] == "circle":
                pygame.draw.circle(self.screen, s2color(cmd.get("color")), self.conv_pos(cmd["origin"]), self.scale * cmd["r"],  width=int(1))

    def flush(self, Hz):
        pygame.display.flip()
        self.clock.tick(Hz)

