import numpy as np
from astropy.coordinates import *
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from astroplan import Observer
from matplotlib import gridspec


# List of all messier numbers
grid = np.arange(0, 110)

# Read In Messier Object locations
Objects = pd.read_csv('Messier.csv', delimiter=',')
Objects.columns = ['RA', 'Dec']

# Convert into list of astropy objects
RaDec = SkyCoord(ra=Objects['RA'], dec=Objects['Dec'])

# Set Personal Location
Loc = EarthLocation(lat='32d13.356m', lon='-110d58.482m', height=9000)
utcoffset = -7*u.hour
t0 = Time('2018-03-24 12:00:00') - utcoffset
Site = Observer(location=Loc)

# Set Cost arguements
ms = 90*u.arcminute  # Slew speed
D = 26.185*u.deg		   # Over/Undershoot correction
A = 10*u.deg      # Altitude bias correction
Obstime = 240

# Bounding Region
A1 = Angle('15d')
A2 = Angle('80d')
Z1 = Angle('225d')
Z2 = Angle('360d')


# Find the location of the sun at given time
def sun_alt(t):
    sun = get_sun(t).transform_to(AltAz(obstime=t, location=Loc))
    return sun.alt


# Find the cost for going bewteem
def cost_f(p0, p1, t):
    # Takes in the current object(p0), target(p1) and current time(t)
    # And returns the cost of jumping between objects and the time it takes

    # find the separation between the current object and target
    sep = p0.separation(p1)

    # get the altitude of target
    _p1 = p1.transform_to(AltAz(obstime=t, location=Loc)).alt

    term1 = (sep/ms).decompose()    # Travel Time
    term2 = (1+(sep/D)**2)          # Over/Under shoot correction term
    term3 = (_p1/A).decompose()  # altitude bias

    # Find the total cost value
    c = float(term1*term2+term3)

    # Find the corresponding time step between objects
    dt = float(term1*term2+Obstime)

    return c, dt


# Choose a random starting object at given time t
# Object must be in chosedn region for it to be chosen
# Current region is set to insure that all objects are observed
def Start(t):
    searching = True
    while searching:
        start = np.random.randint(len(RaDec))
        Object = RaDec[start]
        Object1 = Object.transform_to(AltAz(obstime=t, location=Loc))
        if Object1.alt > A1 and Object1.alt < Angle('30d'):
            if Object1.az > Z1 and Object1.az < Z2:
                searching = False
    return start


def Region_plot():
	# Show the bounding region in the current skyview
    theta = np.arange(-1, 1, 1./180)*np.pi
    r = 90-A1/u.deg+0*theta
    r2 = 90-A2/u.deg+0*theta

    ax1.plot(theta, r2, color='orange')
    ax1.plot(theta, r, color='gold')
    ax1.fill_between(theta, r, r2, alpha=0.3, color='g')

    AA = SkyCoord(theta*u.radian, r*u.deg, frame='altaz', obstime=t, location=Loc)
    AA2 = SkyCoord(theta*u.radian, r2*u.deg, frame='altaz', obstime=t, location=Loc)

    RD = AA.transform_to(FK5)
    RD2 = AA2.transform_to(FK5)

    # Break up and store the coordinates
    RA, DEC = RD.ra.radian, RD.dec.radian
    RA2, DEC2 = RD2.ra.radian, RD2.dec.radian

    # Cycle the region curves to left side of map
    RA = np.remainder(RA+2*np.pi, 2*np.pi)  # shift RA values
    ind = np.where(RA > np.pi)
    RA[ind] -= 2*np.pi   # scale conversion to [-180, 180]
    RA2 = np.remainder(RA2 + 2*np.pi, 2*np.pi)  # shift RA values
    ind = np.where(RA2 > np.pi)[0]
    RA2[ind] -= 2*np.pi   # scale conversion to [-180, 180]

    # Plot the current boundary curves
    ax2.scatter(RA, DEC, marker='.', s=0.5, color='orange')
    ax2.scatter(RA2, DEC2, marker='.', s=0.5, color='gold')


# Plot the observed objects in red circles and the path between them(blue
# lines) in a current skyview(polar plot) and the observed objects in
# fullskyview(mollweide). Also plots all of the Messier objects as black stars
def Plot(Obs, t):
    # Take in the list of observed objects(Obs) at given time(t)

    # Clean the figure so that there are no trails
    ax1.clear()
    ax2.clear()

    # Set up grid linesS
    ax1.grid(b=True, color='k', linestyle='--', alpha=0.3)
    ax2.grid(b=True, color='k', linestyle='--', alpha=0.3)

    # Show time 
    ax1.set_title("t = {}\n".format(t+utcoffset))

    # Project the boundary region
    Region_plot()

    # Transform the objects into AltAz coordinates for the current sky view
    AzAlt = RaDec.transform_to(AltAz(obstime=t, location=Loc))

    # Create arrays to store the objects for plotting
    az, alt = np.zeros(len(Obs)), np.zeros(len(Obs))
    ra, dec = np.zeros(len(Obs)), np.zeros(len(Obs))

    # Store the objects for plotting
    for j in range(len(Obs)):
        az[j] = AzAlt[Obs[j]].az.radian
        alt[j] = 90 - AzAlt[Obs[j]].alt/u.deg
        ra[j] = RaDec[Obs[j]].ra.radian
        if ra[j] > np.pi:
            ra[j] -= 2*np.pi
        dec[j] = RaDec[Obs[j]].dec.radian

    # Plot all the Messier Objects location
    ax1.scatter(AzAlt.az.radian, 90-AzAlt.alt/u.deg, marker='*', Color='k')

    # Plot all the observed objects
    ax1.scatter(az, alt, color='r')

    # Plot the path taken
    ax1.plot(az, alt, linestyle='-', linewidth=1, color='b')

    # Set the axes of the plot
    ax1.set_theta_zero_location('N')
    ax1.set_rlim(0, 90, 1)
    ax1.set_yticks(np.arange(0, 91, 15))
    ax1.set_yticklabels(ax1.get_yticks()[::-1])

    # Plot all of the observed objects
    ax2.scatter(ra, dec, color='r')

    # Get all messier objects
    Ra, Dec = RaDec.ra.radian, RaDec.dec.radian
    Ra = np.remainder(Ra+2*np.pi, 2*np.pi)  # shift RA values
    ind = np.where(Ra > np.pi)
    Ra[ind] -= 2*np.pi   # scale conversion to [-180, 180]

    # Plot all of the Messier Objects
    ax2.scatter(Ra, Dec, marker='*', Color='k')

    # Set the x axis labels for ra coordinates
    ax2.set_xticklabels(['14h', '16h', '18h', '20h',
                         '22h', '0h', '2h', '4h', '6h', '8h', '10h'])
    
    # Set the layout of the plot to tight
    plt.tight_layout()

    # Update and display plot
    plt.draw()
    plt.pause(0.01)


# Run for the the entire night finding the best possible path between
# objects that hopefully allows for observation of all objects
def Run_Through(t, Observed):
    # Takes in the time of starting, the first object, and list that
    # has all of the observed objects(currently only has the first object)
    start = Observed[0]
    # Run while the sun is below the horizon and you still haven't observed
    # every object
    while sun_alt(t) < Angle('0d') and len(Observed) <= 111:
        # Update user on the current state by plotting the objects
        # and printing out the time, current object, and total observed
        Plot(Observed, t)
        plt.savefig("MM_{:003d}.png".format(len(Observed)))
        
        # Show time, object number and howman
        print(t+utcoffset, 'M'+str(start+1), len(Observed))
        f.write("{}, M{}, {}\n".format(t+utcoffset, start+1, len(Observed)))
        # Find the current object and transform to AltAz mode
        Object = RaDec[start]
        Object1 = Object.transform_to(AltAz(obstime=t, location=Loc))

        # Make list of all objects above the lower bound
        _radec = RaDec.transform_to(AltAz(obstime=t, location=Loc)).alt
        Observable = np.where(_radec > A1)[0]

        # Find the object in the Observable list with lowest cost value
        mnm = 1e12  # Initialize lowest cost value with something really high
        for i in range(len(Observable)):
            Object2 = RaDec[Observable[i]]
            cost, dt = cost_f(Object1, Object2, t)
            # Replace the mnm  if lower
            if cost < mnm and Observable[i] not in Observed:
                start = Observable[i]  # set as next object
                mnm = cost
                step = dt  # store step time of object

        # Add object to list if possible and take the time step for that object
        if mnm < 1e12:
            Observed.append(start)
            print(cost, step)
            f.write("{}, {}\n".format(cost, step))
            t += step*u.s
        else:
            # If no object found, add ten minutes
            t += 10*u.min
    # Return the list of Observed objects
    return Observed


# Set up for running program
Final = []  # list to store final list of objects
c = 0       # counter for how many tries

while len(Final) < 110:
    # Run the program until it has observed all objects

    # Start log file to record runs
    f = open('MessierLog.txt', 'w')

    # Set up the plots
    fig = plt.figure(figsize=(8, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    # Set up the current skyview plot
    ax1 = fig.add_subplot(gs[0], projection="polar")

    # Set up the full skyview plot
    ax2 = fig.add_subplot(gs[1], projection="mollweide")

    # Initialize time
    t = Time(Site.sun_set_time(t0, which="next"), format='iso')
    f.write("Start at: {}\n".format(t))

    # Initialize Observed list with starting object
    Observed = [Start(t)]

    # Run the Program
    Final = Run_Through(t, Observed)

    # Add to the counter
    c += 1

    # Get list of all non observed objects
    NO = []
    for g in grid:
        if g not in Final:
            NO.append(g)

    # Print the output of run
    print('\n\n\nTry number:'+str(c))
    f.write('\n\n\nTry number:'+str(c))
    print('\n\n\n Not Observed:\n{}'.format(NO))
    f.write('\n\n\n Not Observed:\n{}'.format(NO))
    print('\n\n\nFinal Length:  '+str(len(Final)))
    f.write('\n\n\nFinal Length:  {}'.format(len(Final)))

    # Close plot
    plt.close()
f.close()
