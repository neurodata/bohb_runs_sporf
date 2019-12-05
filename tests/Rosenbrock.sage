


x, y, a, b = var("x y a b")

a = 1
b = 100

def f(x,y):
    return (a - x)^2 + b*(y - x^2)^2

cm = colormaps.autumn
plot3d(f, (x, -5,5), (y, -5,5), color = "Blues")


sphinx_plot(plot3d(f, (x, -4,4), (y, -4,4)))

P = plot3d(f,(-5,5),(-5,5), adaptive=True, color=rainbow(60, 'rgbtuple'), max_bend=.1, max_depth=15)
P.show()
