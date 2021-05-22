# Python-Vortex-Sim
A simple Python tool to model the behaviour of vortices in the plane.

Functionality so far allows for any number of vortices of varying 'strength' to have their initial positions plotted on the plane, and then a gif is produced of the behaviour of the vortices over time.

Some simple examples can be found below:

![vortexdiagram](https://user-images.githubusercontent.com/79726292/119240588-7f236e80-bb48-11eb-9109-cb676fb66c25.gif)

![vortexdiagram](https://user-images.githubusercontent.com/79726292/119240754-cb22e300-bb49-11eb-9cdf-b1097a1c6bf8.gif)

So far the simulation plots a quiver plot of the direction of flow over a heatmap of the magnitude of the velocities at each point. There is a fair amount of customisability within the arguments, but do be aware there is little protection against singularities, so if you have division by zero error try different bounds so the velocity is never evaluated at the singularity. 

This is quite a simple tool so far, but it is very easy to use, and I plan to add some more functionality to it over the coming weeks.
