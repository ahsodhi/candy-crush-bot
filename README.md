# Candy Crush Bot

This bot reads the the RGB color value from each cell in the gameboard and categorizes it based on manhattan distances.    
It then proceedes to find all possible swaps and the numbers of candies that will be eliminated (including any combos) by performing that swap.    
It executes the swap that will remove the highest number of candies.    
In case there are more than one best possbile swaps, it gives preference to one that's further down the board.  

## Configuration

You will need to adjust the values of the following variables, declared on top of script, to fit your screen size.  

* SNAPSHOT_AREA: area of the screen that will contain the gameboard
* HRANGE: horizontal distance in pixels between each cell of the gameboard
* VRANGE: vertical distance in pixels between each cell of the gameboard
* TOP_LEFT: centre point of the top-left cell in gameboard

## Required Software

Python version 2.7.x  

Python packages  
* mss (2.0.22)
* numpy (1.11.12)
* Pillow (4.0.0)
* pywin32 (220)
* OpenCV-Python

## Instructions

1. Start Candy Crush
2. From a terminal window, execute `python bot.py`
