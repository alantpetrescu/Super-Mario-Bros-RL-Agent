import numpy as np
from matplotlib import pyplot as plt

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class MarioRAMGrid:
    def __init__(self, env):
        self.stompable_enemies = [
            0x00, # Green Koopa
            0x01, 0x03, # Red Koopa
            0x02, # Buzzy Beetle
            0x06, # Goomba
            0x09, # Green Koopa Paratroopa
            0x0A, # Grey Cheep Cheep
            0x0B, # Red Cheep Cheep
            0x0E, # Green Paratroopa Jump
            0x12, # Spiny Egg
            0x14, # Fly Cheep Cheep
            0x37, 0x38, # Little Goomba
            0x3A, # Skewed Goomba
        ]

        self.ram = env.unwrapped.ram
        self.screen_size_x = 16     # rendered screen size
        self.screen_size_y = 13
        
        self.mario_level_x = self.ram[0x6d]*256 + self.ram[0x86]
        self.mario_x_position = self.ram[0x3ad]  # mario's position on the rendered screen
        self.mario_y_position = self.ram[0x3b8] + 16 # top edge of (big) mario
        
        self.x_left = self.mario_level_x - self.mario_x_position # left edge pixel of the rendered screen in level
        self.rendered_screen = self.get_rendered_screen()
    
    def tile_position_to_ram_address(self, x, y):
        '''
        Convert (x, y) in Current tile (32x13, stored as 16x26 in ram) to ram address
        x: 0 to 31
        y: 0 to 12
        This is used to get the background tile grid
        '''
        page = x // 16
        x_position = x%16
        y_position = page*13 + y
        
        address = 0x500 + x_position + y_position*16
        
        return address

    def get_rendered_screen(self):
        '''
        Get the rendered screen (13 x 16) from RAM
        empty: 0
        tile: 1
        mario: 2
        stompable enemy: -1
        unstompable enemy: -2
        '''
        
        # Get background tiles
        
        rendered_screen = np.zeros((self.screen_size_y, self.screen_size_x))
        screen_left = int(np.rint(self.x_left / 16))
    
        for i in range(self.screen_size_x):
            for j in range(self.screen_size_y):
                x_position = (screen_left + i) % (self.screen_size_x * 2)
                y_position = j
                address = self.tile_position_to_ram_address(x_position, y_position) 

                # Convert all types of tile to 1
                if self.ram[address] != 0:
                    rendered_screen[j, i] = 1
                    
        # Add mario
        x_position = (self.mario_x_position + 8) // 16
        y_position = (self.mario_y_position - 32) // 16 # top 2 rows in the rendered screen aren't stored in ram
        if x_position < 16 and y_position < 13:
            rendered_screen[y_position, x_position] = 2
        
        # Add enemies
        for i in range(5):
            # check if the enemy is drawn
            if self.ram[0xF + i] == 1: 
                enemy_x_position = self.ram[0x6e + i]*256 + self.ram[0x87 + i] - self.x_left
                enemy_y_position = self.ram[0xcf + i]
                x_position = (enemy_x_position + 8) // 16
                y_position = (enemy_y_position + 8 - 32) // 16

                # check if the enemy is inside the rendered screen
                if 0 <= x_position < 16 and 0 <= y_position < 13:
                    if self.ram[0x16 + i] in self.stompable_enemies:
                        rendered_screen[y_position, x_position] = -1 # The enemy can be stomped
                    else:
                        rendered_screen[y_position, x_position] = -2 # The enemy can't be stomped
                
        return rendered_screen