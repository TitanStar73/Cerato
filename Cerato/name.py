#New version  opener  to celebrate v1.0.0

text = r"""     𓅪                                                       
  /$$$$$$                                 /$$              
 /$$__  $$                               | $$              
| $$  \__/  /$$$$$$   /$$$$$$  /$$$$$$  /$$$$$$    /$$$$$$ 
| $$       /$$__  $$ /$$__  $$|____  $$|_  $$_/   /$$__  $$
| $$      | $$$$$$$$| $$  \__/ /$$$$$$$  | $$    | $$  \ $$
| $$    $$| $$_____/| $$      /$$__  $$  | $$ /$$| $$  | $$
|  $$$$$$/|  $$$$$$$| $$     |  $$$$$$$  |  $$$$/|  $$$$$$/
 \______/  \_______/|__/      \_______/   \___/   \______/  

""".replace("$","#")

def gradient_text(text, allowed_chars, colors):
    """
    text: multiline string
    allowed_chars: set of characters to color
    colors: list of (r, g, b) tuples defining gradient stops
    """
    lines = text.splitlines()
    height = len(lines)
    width = max(len(line) for line in lines) if lines else 0
    if width == 0 or not colors:
        return text

    def interp_color(t):
        """Interpolate color from multiple gradient stops."""
        if t <= 0:
            return colors[0]
        if t >= 1:
            return colors[-1]
        segment = (len(colors) - 1) * t
        i = int(segment)
        frac = segment - i
        r1, g1, b1 = colors[i]
        r2, g2, b2 = colors[i + 1]
        return (
            int(r1 + frac * (r2 - r1)),
            int(g1 + frac * (g2 - g1)),
            int(b1 + frac * (b2 - b1)),
        )

    result = []
    for y, line in enumerate(lines):
        for x, ch in enumerate(line.ljust(width)):
            if ch in allowed_chars:
                m = 2.65
                t = ((m*x / max(1, width - 1)) + (y / max(1, height - 1))) / (1+m)
                
                r, g, b = interp_color(t)
                result.append(f"\033[38;2;{r};{g};{b}m{ch}\033[0m")
            else:
                result.append(ch)
        result.append("\n")

    return "".join(result)


allowed = set("#╔╗ ╦ ╦  ┌─┐┬─┐┬ ┬┌─┐┬ ┬╠╩╗╚╦╝  ├─┤├┬┘│ │└─┐├─┤╚═╝ ╩   ┴ ┴┴└─└─┘└─┘┴ ┴𓅪")

rainbow_colors = [
    (255, 0, 0),       # Red
    (255, 100, 0),     # Orange
    (255, 220, 0),      # Yellow with slightly toned-down saturation (looked to contrasted on terminal)
    (0, 200, 0),       # Green
    (0, 80, 255),      # Blue
    (120, 0, 255),     # Indigo
    (255, 0, 255),      # Violet
]

weights = [3,3,2,2,3,2,2]
rainbow = []

for ind, col in enumerate(rainbow_colors):
    for i in range(weights[ind]):
        rainbow.append(col)

print(gradient_text(text, allowed, rainbow))
