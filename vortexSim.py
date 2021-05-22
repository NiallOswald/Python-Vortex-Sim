class VortexSim():
    'An analytic vortex simulation for n vortices of set strengths'

    # Global imports
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    def __init__(self, vortex_points, x1, y1, x2, y2, strengths = None, *, step = 20):
        # Initial setup

        ## Setting the coordinates of each vortex
        self.vortex_points = vortex_points

        ## Setting the strength of each vortex
        if strengths == None:
            self.strengths = [1 for i in vortex_points]
        else:
            self.strengths = strengths

        ## Setting plotting dimensions for the heatmap
        self.extent = [x1, x2, y1, y2]

        ## Setting the ratio of heatmap points to vectors
        self.step = step

        ## Fixing points for the heatmap
        x_values = self.np.linspace(x1, x2, 100*(x2 - x1))
        y_values = self.np.linspace(y1, y2, 100*(y2 - y1))

        self.x, self.y = self.np.meshgrid(x_values, y_values)

        ## Fixing locations of each vector
        self.xq = [i[::step] for i in self.x[::step]]
        self.yq = [i[::step] for i in self.y[::step]]

        ## Generating initial velocity data for heatmap
        velocities = [self.update_velocities(self.x, self.y, i[1][0], i[1][1], self.strengths[i[0]]) for i in enumerate(vortex_points)]

        dxdt, dydt = 0, 0
        for i in velocities:
            dxdt += i[0]
            dydt += i[1]
            
        self.v = self.np.sqrt(dxdt**2 + dydt**2)

        ## Generating initial velocity data for quiver plot
        self.dxdtq = [i[::step] for i in dxdt[::step]]
        self.dydtq = [i[::step] for i in dydt[::step]]

    def save_sim(self, length = 100, interval = 50):
        'Saves a gif of the vortex simulation to filepath'
        from astropy.visualization import (MinMaxInterval, LogStretch, ImageNormalize)

        # Setting up the plot
        self.fig, self.ax = self.plt.subplots(1, 1)

        # Normalising the heatmap colours using a log stretch
        norm = ImageNormalize(self.v, interval=MinMaxInterval(), stretch=LogStretch())

        # Plotting the initial heatmap
        self.im = self.ax.imshow(self.v, origin='lower', norm=norm, extent=self.extent)

        # Plotting the initial quiver plot
        self.Q = self.ax.quiver(self.xq, self.yq, self.dxdtq, self.dydtq, pivot='mid')

        package = [self.im, self.Q]

        # Animating the movement of the quiver plot
        animator = self.FuncAnimation(self.fig, self.update_plots, fargs = (self.x, self.y, self.strengths, self.step), frames = length, interval = interval, blit=False)
        self.fig.tight_layout()

        # Saving gif of animation
        animator.save('vortexdiagram.gif', writer='imagemagick')

    def update_velocities(self, x, y, x_vortex, y_vortex, strength):
        'Finds velocities of a single vortex, when the vortex is translated away from the origin'
        
        # Translated point field
        xt = x - x_vortex
        yt = y - y_vortex
        
        # Velocities
        dxdt = -strength*yt/(xt**2 + yt**2)
        dydt = strength*xt/(xt**2 + yt**2)
        
        return (dxdt, dydt)

    def move_vortex(self, vortex_points, movable, strengths, *, delta = 0.05):
        'Finds the velocity vector acting on a vortex, when that vortex is removed'
        
        # Creates a copy of vortex_points and then pops the movable vortex
        vortex_fixed = vortex_points.copy()
        vortex_movable = vortex_fixed.pop(movable)
        
        xm, ym = vortex_movable[0], vortex_movable[1]
        
        # Removing the strength of the movable vortex, as to prevent iteration problems
        fixed_strengths = strengths.copy()
        fixed_strengths.pop(movable)
        
        # Defining lambda expressions to calculate the derivative at xm, ym for each fixed vortex
        x_deriv = lambda xf, yf, strength: -strength*(ym - yf)/((xm - xf)**2 + (ym - yf)**2)
        y_deriv = lambda xf, yf, strength: strength*(xm - xf)/((xm - xf)**2 + (ym - yf)**2)
        
        # Sums the derivatives of each fixed vortex to find the new location of the movable vortex
        dxdt, dydt = 0, 0
        for i in enumerate(vortex_fixed):
            dxdt += x_deriv(i[1][0], i[1][1], fixed_strengths[i[0]])
            dydt += y_deriv(i[1][0], i[1][1], fixed_strengths[i[0]])
        
        return (xm + delta*dxdt, ym + delta*dydt)

    def update_plots(self, num, x, y, strengths, step):
        'Updates the quiver plot for each tick of time'
        
        updated_vortex_points = [self.move_vortex(self.vortex_points, i, strengths) for i in range(len(self.vortex_points))]
        
        self.vortex_points = updated_vortex_points.copy()
        
        velocities = [self.update_velocities(x, y, i[1][0], i[1][1], strengths[i[0]]) for i in enumerate(self.vortex_points)]
        
        dxdt, dydt = 0, 0
        for i in velocities:
            dxdt += i[0]
            dydt += i[1]
        
        v = self.np.sqrt(dxdt**2 + dydt**2)
        
        dxdtq = [i[::step] for i in dxdt[::step]]
        dydtq = [i[::step] for i in dydt[::step]]
        
        self.Q.set_UVC(dxdtq, dydtq)
        self.im.set_array(v)
        
        return [self.im, self.Q]