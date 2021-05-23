class VortexSim():
    'An analytic vortex simulation for n vortices of set strengths'

    # Class-wide imports
    import numpy as np

    def __init__(self, vortex_points, x1, y1, x2, y2, strengths = None, damping = 0.0, *, step = 20):
        """
        Parameters:

            vortex_points (list): A list of tuples of length 2 containing the coordinates of the vortices

            x1, y1, x2, y2 (float): Floating point numbers representing the bounds of the plane that will be plotted

            strengths (list): A list of floats which represent the strength of the vortex. A negative strength is equivalent to a negatively signed vortex

            damping (float): A non-negative float between 0 and 1, which is used to reduce the strengths of the vortices at each step of time

            step (int): An integer representing the ratio of points in the heatmap to vectors plotted in the quiver plot
        """

        # Initial setup

        ## Setting the coordinates of each vortex
        self.vortex_points = vortex_points

        ## Setting the strength of each vortex
        if strengths == None:
            self.strengths = [1 for i in vortex_points]
        else:
            self.strengths = strengths

        ## Setting the damping factor of each vortex
        self.damping = damping

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
        """
        Saves a gif of the vortex simulation to filepath

        Parameters:

            length (int): The total number of frames present in the gif

            interval (int): The number of milliseconds between frames in the gif
        """

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from astropy.visualization import (MinMaxInterval, LogStretch, ImageNormalize)

        # Setting up the plot
        self.fig, self.ax = plt.subplots(1, 1)

        # Normalising the heatmap colours using a log stretch
        norm = ImageNormalize(self.v, interval=MinMaxInterval(), stretch=LogStretch())

        # Plotting the initial heatmap
        self.im = self.ax.imshow(self.v, origin='lower', norm=norm, extent=self.extent)

        # Plotting the initial quiver plot
        self.Q = self.ax.quiver(self.xq, self.yq, self.dxdtq, self.dydtq, pivot='mid')

        # Animating the movement of the quiver plot
        animator = FuncAnimation(self.fig, self.update_plots, fargs = (self.x, self.y, self.damping, self.step), 
                                    frames = length, interval = interval, blit=False)
        self.fig.tight_layout()

        # Saving gif of animation
        animator.save('vortexdiagram.gif', writer='imagemagick')

    def update_velocities(self, x, y, x_vortex, y_vortex, strength):
        'Finds velocities of a single vortex, when the vortex is translated away from the origin'
        
        # Translated point field
        xt = x - x_vortex
        yt = y - y_vortex
        
        # Generate new velocities
        dxdt = -strength*yt/(xt**2 + yt**2)
        dydt = strength*xt/(xt**2 + yt**2)
        
        # Return new velocities
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
        
        # Return updated vortex position
        return (xm + delta*dxdt, ym + delta*dydt)

    def update_plots(self, num, x, y, damping, step):
        'Updates the plots for each tick of time'

        # Update strengths with damping factor
        self.strengths = [x*(1 - damping) for x in self.strengths]
        
        # Update the positions of the vortices
        updated_vortex_points = [self.move_vortex(self.vortex_points, i, self.strengths) for i in range(len(self.vortex_points))]
        
        self.vortex_points = updated_vortex_points.copy()
        
        # Find the velocities generated by each vortex singularly
        velocities = [self.update_velocities(x, y, i[1][0], i[1][1], self.strengths[i[0]]) for i in enumerate(self.vortex_points)]
        
        # Sum to find the total velocities induced by all vortices
        dxdt, dydt = 0, 0
        for i in velocities:
            dxdt += i[0]
            dydt += i[1]
        
        # Generate a scalar speed by taking the absolute value of all the velocity vectors
        v = self.np.sqrt(dxdt**2 + dydt**2)
        
        # Create a lower resolution list of velocities for use in the quiver plot
        dxdtq = [i[::step] for i in dxdt[::step]]
        dydtq = [i[::step] for i in dydt[::step]]
        
        # Update plot data
        self.Q.set_UVC(dxdtq, dydtq)
        self.im.set_array(v)
        
        # Return updated plots
        return [self.im, self.Q]
    
class VortexStreet():
    'A pseudo-simulation for a vortex street'

    # Class-wide imports
    import numpy as np

    def __init__(self, generation_points, period, underlying_velocity, *, dimensions = ((-2, -2), (2, 2)), damping = 0.0, step = 20):
        """
        Parameters:

            generation_points (tuple): A tuple of tuples of length 2, containing the coordinates of two points where vortices are generated periodically

            period (int): An integer representing the number of frames between vortices generated

            underlying_velocity (tuple): A tuple of two lambda functions which give the constant underlying vector field

            dimensions (tuple): A tuple of length 2 containing the coordinates of the corners of the plot over which the vectors are evaluated as tuples

            damping (float): A non-negative float between 0 and 1, which is used to reduce the strengths of the vortices at each frame

            step (int): An integer representing the ratio of points in the heatmap to vectors plotted in the quiver plot
        """

        # Initial setup

        ## Setting points where vorticies can be generated
        self.generation_points = generation_points

        ## Setting the number of frames between vortices generated
        self.period = period

        ## Setting the underlying vector field as lambda functions
        self.dxdtf, self.dydtf = underlying_velocity[0], underlying_velocity[1]

        ## Setting the dimensions of the plot
        self.x1, self.y1, self.x2, self.y2 = dimensions[0][0], dimensions[0][1], dimensions[1][0], dimensions[1][1]

        ## Setting the damping factor
        self.damping = damping

        ## Setting the ratio of heatmap points to vectors
        self.step = step

        ## Setting the coordinates of the initial vortex
        self.vortex_points = [generation_points[0]]

        ## Setting the strength of the initial vortex
        self.strengths = [-1]

    def new_vortex(self, i, strength = 1):
        'Adds a new vortex at one of the generation point periodically'

        self.place_vortex(self.generation_points[i % 2], strength)

    def place_vortex(self, coords, strength):
        'Adds a new vortex at the given points'

        # Updates vortex_points
        self.vortex_points.append((coords[0], coords[1]))

        # Updates strengths
        self.strengths.append(strength)

    def update_velocities(self, x, y, x0, y0, strength):
        'Finds velocities of a single vortex, when the vortex is translated away from the origin'
        
        # Translated point field
        xt = x - x0
        yt = y - y0
        
        # Generate new velocities
        dxdt = -strength*yt/(xt**2 + yt**2)
        dydt = strength*xt/(xt**2 + yt**2)
        
        # Return new velocities
        return (dxdt, dydt)

    def move_vortex(self, movable, delta):
        'Finds the velocity vector acting on a vortex, when that vortex is removed'
        
        # Creates a copy of vortex_points and then pops the movable vortex
        vortex_fixed = self.vortex_points.copy()
        vortex_movable = vortex_fixed.pop(movable)
        
        xm, ym = vortex_movable[0], vortex_movable[1]
        
        # Removing the strength of the movable vortex, as to prevent iteration problems
        fixed_strengths = self.strengths.copy()
        fixed_strengths.pop(movable)
        
        # Defining lambda expressions to calculate the derivative at xm, ym for each fixed vortex
        x_deriv = lambda xf, yf, strength: -strength*(ym - yf)/((xm - xf)**2 + (ym - yf)**2)
        y_deriv = lambda xf, yf, strength: strength*(xm - xf)/((xm - xf)**2 + (ym - yf)**2)
        
        # Sets the initial values to the underlying vector field at (xm, ym)
        dxdt, dydt = self.dxdtf(xm, ym), self.dydtf(xm, ym)

        # Sums the derivatives of each fixed vortex to find the new location of the movable vortex
        for i in enumerate(vortex_fixed):
            dxdt += x_deriv(i[1][0], i[1][1], fixed_strengths[i[0]])
            dydt += y_deriv(i[1][0], i[1][1], fixed_strengths[i[0]])
        
        # Return updated vortex position
        return (xm + delta*dxdt, ym + delta*dydt)

    def update_plots(self, num, x, y, delta):
        'Updates the plots for each frame'

        # Determine whether a new vortex is to be placed
        if (num % self.period == 0) and (num != 0):
            self.new_vortex(num//self.period, (-1)**(num//self.period + 1))

        # Update strengths with damping factor
        self.strengths = [x*(1 - self.damping) for x in self.strengths]
        
        # Update the positions of the vortices
        updated_vortex_points = [self.move_vortex(i, delta) for i in range(len(self.vortex_points))]
        
        self.vortex_points = updated_vortex_points.copy()
        
        # Find the velocities generated by each vortex singularly
        velocities = [self.update_velocities(x, y, i[1][0], i[1][1], self.strengths[i[0]]) for i in enumerate(self.vortex_points)]
        
        # Sets the initial values of the underlying vector field
        dxdt, dydt = self.dxdtf(x, y), self.dydtf(x, y)

        # Sum to find the total velocities induced by all vortices
        for i in velocities:
            dxdt += i[0]
            dydt += i[1]
        
        # Generate a scalar speed by taking the absolute value of all the velocity vectors
        v = self.np.sqrt(dxdt**2 + dydt**2)
        
        # Create a lower resolution list of velocities for use in the quiver plot
        dxdtq = [i[::self.step] for i in dxdt[::self.step]]
        dydtq = [i[::self.step] for i in dydt[::self.step]]
        
        # Update plot data
        self.Q.set_UVC(dxdtq, dydtq)
        self.im.set_array(v)
        
        # Return updated plots
        return [self.im, self.Q]

    def save_sim(self, *, frames = 100, interval = 50, delta = 0.05):
        """
        Saves a gif of the vortex simulation to filepath

        Parameters:

            length (int): The total number of frames present in the gif

            interval (int): The number of milliseconds between frames in the gif

            delta (float): A float representing the size of steps between each frame
        """

        ## Fixing points for the heatmap
        x_values = self.np.linspace(self.x1, self.x2, 100*(self.x2 - self.x1))
        y_values = self.np.linspace(self.y1, self.y2, 100*(self.y2 - self.y1))

        x, y = self.np.meshgrid(x_values, y_values)

        ## Fixing locations of each vector
        xq = [i[::self.step] for i in x[::self.step]]
        yq = [i[::self.step] for i in y[::self.step]]

        ## Generating initial velocity data for heatmap
        velocities = [self.update_velocities(x, y, i[1][0], i[1][1], self.strengths[i[0]]) for i in enumerate(self.vortex_points)]

        dxdt, dydt = self.dxdtf(x, y), self.dydtf(x, y)
        for i in velocities:
            dxdt += i[0]
            dydt += i[1]
            
        v = self.np.sqrt(dxdt**2 + dydt**2)

        ## Generating initial velocity data for quiver plot
        dxdtq = [i[::self.step] for i in dxdt[::self.step]]
        dydtq = [i[::self.step] for i in dydt[::self.step]]

        ####################

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from astropy.visualization import (MinMaxInterval, LogStretch, ImageNormalize)

        # Setting up the plot
        self.fig, self.ax = plt.subplots(1, 1)

        # Normalising the heatmap colours using a log stretch
        norm = ImageNormalize(v, interval = MinMaxInterval(), stretch = LogStretch())

        ## Setting plotting dimensions for the heatmap
        extent = [self.x1, self.x2, self.y1, self.y2]

        # Plotting the initial heatmap
        self.im = self.ax.imshow(v, origin = 'lower', norm = norm, extent = extent)

        # Plotting the initial quiver plot
        self.Q = self.ax.quiver(xq, yq, dxdtq, dydtq, pivot = 'mid')

        # Animating the movement of the quiver plot
        animator = FuncAnimation(self.fig, self.update_plots, fargs = (x, y, delta), frames = frames, interval = interval, blit = False)
        self.fig.tight_layout()

        # Saving gif of animation
        animator.save('vortexstreet.gif', writer='imagemagick')
