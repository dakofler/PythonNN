from library import helpers as hlp
import math

def plot_color(input):
    r = (input[0] / 2 + 0.5) * 255
    g = (input[1] / 2 + 0.5) * 255
    b = (input[2] / 2 + 0.5) * 255
    hlp.printmd(f'<div style="height: 100px; width: 100px; background-color: rgb({r},{g},{b});"></div>')

def rgb_to_norm(rgb):
    r_norm = round((rgb[0] / 255 - 0.5) * 2, 2)
    g_norm = round((rgb[1] / 255 - 0.5) * 2, 2)
    b_norm = round((rgb[2] / 255 - 0.5) * 2, 2)
    return [r_norm, g_norm, b_norm]