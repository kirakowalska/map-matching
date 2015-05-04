import numpy as np
import random
from sets import Set
import math
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# Classes

# In[11]:

class Point:   # No spatial contraints.
    # Class for storing both GPS points and particles
    def __init__(self, E, N, speed, bearing, datetime):
        self.e = E
        self.n = N
        self.s = speed
        self.b = bearing
        self.dt = datetime


# In[12]:

class Particle:  # Limited to street network.
    def __init__(self, fid, distance_from_start, orientation, speed, journey_history):
        self.fid = fid
        self.d = distance_from_start    # distance from start node of the segment ('orientation_neg') --> not indicative of the direction
                                        # of traffic flow
        self.o = orientation    # 'pos' or 'neg'
        self.s = speed
        self.h = journey_history
    """
    def __init__(self, E, N, speed, bearing, fid, fid_sec):
        self.e = E
        self.n = N
        self.s = speed
        self.b = bearing
        self.fid = fid
        self.fid_sec = fid_sec    # numerical: indicates which section of the segment the particle is on
    """

    
# Functions

# In[13]:

def initialize_particles(n,GPS_first):

    particles = []    # List of particles (Particle, weight)
    weights = []

    for i in range(0,n):
        # 1. Sample particle
        particle = sample_particle_with_noise(GPS_first,10)
        particles.append(particle)
        weights.append(1.0)

    # Normalize weights
    weights = [i/sum(weights) for i in weights]

    S = zip(particles, weights)
    return S


# In[14]:

def initialize_particles_network(n,GPS_first,street_network):
    
    # Extract starts and ends of each segment
    edge_polylines = street_network[2]
    segments = []
    seg_starts = []
    seg_ends = []
    for key, value in edge_polylines.iteritems():
        segments.append(key)
        seg_starts.append(value[0,:])
        seg_ends.append(value[-1,:])
    seg_starts = np.asarray(seg_starts)
    seg_ends = np.asarray(seg_ends)
    
    # Initialise particles
    particles = []
    weights = []

    iter_no = 0

    local_inds1 = close_pass_filter(seg_starts, GPS_first, d=50)
    local_inds2 = close_pass_filter(seg_ends, GPS_first, d=50)
    segments_local = [segments[i] for i in list(Set(np.concatenate((local_inds1,local_inds2))))]
    camden_network = street_network[0]

    no_particles = 0

    while no_particles < n:
        iter_no += 1
        # Sample point
        point = sample_point_with_noise(GPS_first,10)

        # Check if point is on a segment
        for segment_fid in segments_local:
            segment = edge_polylines[segment_fid]
            distance_from_start = 0
            for i in range(1,len(segment)):
                segment_orientation = camden_network.es.select(fid_eq = segment_fid)[0]['one_way']
                is_on_segment, orientation, distance = is_point_moving_along_segment(segment[i-1],segment[i],segment_orientation, point,30) # epsilon = 10
                #is_on_segment, distance = is_point_on_segment(segment[i-1],segment[i],point,5) # epsilon = 10
                #orientation = 'pos'

                distance_from_start += distance
                if is_on_segment:
                    particle = Particle(fid = segment_fid, distance_from_start = distance_from_start, orientation = orientation, speed = point.s, journey_history = [segment_fid])

                    no_particles +=1
                    particles.append(particle)
                    weights.append(1.0)
                    break
                    break

    # Normalize weights
    weights = [i/sum(weights) for i in weights]

    S = zip(particles, weights)
    return S, iter_no


# In[15]:

def degrees_to_radians(angle_degrees):
    angle_radians = float(angle_degrees)*np.pi/180
    return angle_radians


# In[16]:

def distance_between_points(a,b):
    dist = math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))
    return dist


# In[17]:

def particle_filter(S,u,z, street_network):
    """
    S: a list of particles and their weights [(s_0,w_0),(s_1,w_1),...,(s_n,w_n)]
    u: time passed from last measurement as datetime.timedelta object
    z: current GPS reading
    """
    particles = []
    x_base = []
    displacements = []
    weights = []
    sigma_dist = 10  # affects particle updates, but not particle initialization (which is set to 5m tolerance)

    mu = 0
    n = len(S)  # number of particles

    # Diagnostics: sampling outcome
    samples = []

    for i in range(0,n):
        x,i = weighted_choice(S)
        samples.append(i)
        x_new, displacement = update_particle_network(x,u,sigma_dist, street_network)
        x_base.append((x,displacement)) # store sampled particles + displacement applied (DELETE LATEER)
        displacements.append(displacement)

        #x_new = update_particle(x,u,sigma)
        #w_new = update_weight(x_new,z,sigma)
        w_new = update_weight_network(x_new,z,sigma_dist,street_network)
        particles.append(x_new)
        weights.append(w_new)

        if sum(weights)==0:
            raise Exception("Particles are too unlikely, increase sigma_dist.")

    # Normalize weights
    weights = [i/sum(weights) for i in weights]

    S_new = zip(particles, weights)
    return S_new, x_base, np.mean(displacements)


# In[18]:

def update_particle(x,u, sigma):
    """
    Update particle x in response to control u. In our case, u is time passed (datetime.timedelta object).
    """

    # Get time passed in hours
    t = float(u.seconds)/3600

    # Get bearing in radians
    a = degrees_to_radians(x.b)

    # Update position
    e_new = x.e + t*x.s*np.cos(a)
    n_new = x.n + t*x.s*np.sin(a)

    # Updated mean particle
    x_control = x
    x_control.e = e_new
    x_control.n = n_new

    # Sample new particle with added independent Gaussian noise
    x_new = sample_particle_with_noise(x_control,sigma)

    return x_new


# In[19]:

def get_segment_length(segment_fid, street_network):
    """
    Output segment length calculated from segment edges (a bit more accurate than using the edge attribute 'length').
    """
    
    # Unpack street network
    edge_polylines = street_network[2]
    
    length_total = 0
    segment = edge_polylines[segment_fid]
    for i in range(1,len(segment)):
        length_total += distance_between_points(segment[i-1],segment[i])
    return length_total


# In[20]:

def update_particle_network(x,u,sigma_dist, street_network, print_on = False):
    """
    Update particle x in response to control u. In our case, u is time passed (datetime.timedelta object).
    """

    # Get time passed in hours
    t = float(u.seconds)/3600

    # Get distance travelled estimate
    distance = t*x.s

    # Updated mean particle
    x_control = move_particle(x,distance, street_network)

    if print_on:
        print "Input particle journey"
        print x.fid
        print x.h

        print "Displaced particle journey"
        print x_control.fid
        print x_control.h

    # Sample new particle with added independent Gaussian noise
    x_new, displacement = sample_particle_with_noise_network(x_control,sigma_dist,street_network)

    if print_on:
        print "Displaced noisy particle journey"
        print x_new.fid
        print x_new.h

    if x_new.d <0:
        print x_new.d

    return x_new, displacement


# In[21]:

def get_particle_e_n(particle,street_network):
    """
    Retrieve easting and northing of a particle, given segment fid and distace along that segment.
    """
    
    # Unpack street network
    edge_polylines = street_network[2]
    camden_network = street_network[0]
    
    segment = edge_polylines[particle.fid]

    # Check if distance along segment is smaller than or equal to segment length
    if particle.d > get_segment_length(particle.fid, street_network):
        print "Particle outside the segment assigned by that many meters:"
        print particle.d - camden_network.es.select(fid_eq = particle.fid)[0]['length']

    # Iterate through segment until you reach subsegment on which particle is placed.
    distance_total = 0
    i = 0
    while distance_total <= particle.d:
        # If particle coincides with the end of the subsegment, stop.
        if distance_total == particle.d:
            distance = 0
            break
        # If not, calculate length of the next subsegment and add it to the total traversed distance.
        else:
            i += 1
            distance = distance_between_points(segment[i-1],segment[i])
            distance_total += distance


    # Get coordinates using triangles proportionality theorem
    # If particle coincides with the end of the subsegment
    if distance == 0:
        e = segment[i][0]
        n = segment[i][1]
    # Else, use triangle proportionality theorem to get e and n for the particle.
    else:
        d = distance_total - particle.d
        D = distance
        N_diff = segment[i][1] - segment[i-1][1]
        E_diff = segment[i][0] - segment[i-1][0]

        e_diff = (E_diff*d)/D
        n_diff = (N_diff*d)/D

        e = segment[i][0] - e_diff
        n = segment[i][1] - n_diff

    return e,n


# In[22]:

def get_particle_bearing(particle, street_network):
    
    # Unpack street network
    edge_polylines = street_network[2]

    # Go down to the subsegment that the particle is on.
    segment = edge_polylines[particle.fid]
    distance_from_start = 0
    for i in range(1,len(segment)):
        distance_from_start += distance_between_points(segment[i-1],segment[i])
        if distance_from_start >= particle.d:
            break

    # Depending on the orientation of travel, get bearing.
    if particle.o == 'pos':
        b = get_segment_bearing(segment[i-1],segment[i])
    elif particle.o == 'neg':
        b = get_segment_bearing(segment[i],segment[i-1])

    return b


# In[23]:

def update_weight(pa,po,sigma):
    """
    Get weight of particle pa given the observed GPS point po.
    """
    e = norm(po.e, sigma).pdf(pa.e)
    n = norm(po.n, sigma).pdf(pa.n)
    s = norm(po.s, sigma).pdf(pa.s)
    b = norm(po.b, sigma).pdf(pa.b)

    return e*n*s*b


# In[24]:

def update_weight_network(pa,po,sigma, street_network):
    """
    Get weight of particle pa given the observed GPS point po.
    """
    e_pa, n_pa = get_particle_e_n(pa, street_network)
    b_pa = get_particle_bearing(pa, street_network)

    e = norm(po.e, sigma).pdf(e_pa)
    n = norm(po.n, sigma).pdf(n_pa)
    b = norm(po.b, sigma).pdf(b_pa)
    s = norm(po.s, sigma).pdf(pa.s)

    return e*n*b*s


# In[25]:

def sample_point_with_noise(point,sigma):
    """
    Saple point_new from a Gaussian distribution centered at point with standard deviation sigma.
    """
    # Sample E
    e = random.gauss(point.e, sigma)
    # Sample N
    n = random.gauss(point.n, sigma)
    # Sample speed
    s = random.gauss(point.s, sigma)
    # Sample bearing
    b = random.gauss(point.b, sigma)

    return Point(e,n,s,b,np.nan)


# In[26]:

def move_particle(x, distance, street_network,print_on = False):
    """
    Move particle x along the street network by the given distance. Return updated particle and journey travelled (in terms of
    street segments traversed). The method assumes that particle's orientation is in agreement
    with the street (e.g. not against the flow of traffic on one-way streets.)
    """
    """
    print "Initial fid"
    print x.fid
    """
    
    # Unpack street network
    camden_network = street_network[0]
    

    # Check if particle's orientation is in agreement with the segment
    if not validate_particle_orientation(x,street_network):
        raise Exception("Particle's orientation invalid.")

    # Update fid, orientation, journey history and distance along segment
    if x.o == 'pos':
        # Check if particle will move outside of current segment:
        distance_left_on_segment = get_segment_length(x.fid, street_network) - x.d
    elif x.o == 'neg':
        # Check if particle will move outside of current segment:
        distance_left_on_segment = x.d
    # Initialize journey history
    x_h_new = list(x.h)
    h_new = []

    # No need to leave the segment
    if distance_left_on_segment >= distance:
        fid_new = x.fid
        o_new = x.o
        if x.o == 'pos':
            d_new = x.d + distance
        if x.o == 'neg':
            d_new = x.d - distance
    # Need to venture out
    else:
        distance_travelled = distance_left_on_segment
        fid_new = x.fid
        o_new = x.o
        while distance_travelled <= distance:
            # Get candidate next segments
            node_id = camden_network.es.select(fid_eq = fid_new)['orientation_' + o_new][0]
            segment_fids = camden_network.es(camden_network.adjacent(camden_network.vs.select(id_eq = node_id)[0]))['fid']
            segments_possible = set(segment_fids) - set([fid_new])
            # Randomly select next segment
            # If there are possible next segments
            if len(list(segments_possible))>0:
                fid_new = random.choice(list(segments_possible))
                # Update journey history
                h_new.append(fid_new)

            # No new segments are possible, but you can still go backwards.
            elif fid_new in set(segment_fids):
                fid_new = fid_new # unchanged

                """
                # (Already taken care of below) In that case, change orientation of travel
                if x.o == 'pos':
                    x.o = 'neg'
                if x.o == 'neg':
                    x.o = 'pos'
                """
            # No new segments, and the old one does not allow going backwards (one-way)
            else:
                print "Dead end"
                d_new = seg_length
                x_h_new.extend(h_new)

                particle_new = Particle(fid=fid_new, distance_from_start = d_new, orientation = o_new, speed = x.s, journey_history = x_h_new)
                return particle_new

            seg_length = get_segment_length(fid_new, street_network)
            distance_travelled += seg_length
            # Update orientation of travel
            o_new = update_orientation(node_id, fid_new,street_network)

        """ TAKEN OUT TODAY
        ## NEW
        # Update orientation
        o_new = update_orientation(node_id, fid_new)
        """

        if o_new == 'pos':
            d_new = distance - (distance_travelled - seg_length)
        if o_new == 'neg':
            d_new = seg_length - (distance - (distance_travelled - seg_length))
        ## NEW END

        """
        if x.o == 'pos':
            d_new = distance - (distance_travelled - seg_length)
            # Update orientation
            if camden_network.es.select(fid_eq = fid_new)['orientation_neg'][0] == node_id:
                o_new = 'pos'
            elif camden_network.es.select(fid_eq = fid_new)['orientation_pos'][0] == node_id:
                o_new = 'neg'
            else:
                print "Something went wrong."
        if x.o == 'neg':
            d_new = seg_length - (distance - (distance_travelled - seg_length))
            # Update orientation
            if camden_network.es.select(fid_eq = fid_new)['orientation_neg'][0] == node_id:
                o_new = 'pos'
            elif camden_network.es.select(fid_eq = fid_new)['orientation_pos'][0] == node_id:
                o_new = 'neg'
            else:
                print "Something went wrong."
        """
    """
    if len(h_new)>0:
        print h_new[0]
        print x.h
        if h_new[0]==x.h[-1]:
            for item in x.h:
                print item
            for item in h_new:
                print item
    """

    x_h_new.extend(h_new)


    """
    # Check if particle's history makes sense
    if fid_new != x_h_new[-1]:
        print "Output mismatch"
        print fid_new
        print x_h_new
    """

    # Updated particle
    particle_new = Particle(fid=fid_new, distance_from_start = d_new, orientation = o_new, speed = x.s, journey_history = x_h_new)
    if print_on:
        print particle_new.h

    return particle_new


# In[27]:

def update_orientation(node_visited, fid,street_network):
    """
    Return orientation of travel ('pos' or 'neg') of a particle given the last visited node and street segment the particle is on.
    """
    
    # Unpack street network
    camden_network = street_network[0]

    if camden_network.es.select(fid_eq = fid)['orientation_neg'][0] == node_visited:
        o_new = 'pos'
    elif camden_network.es.select(fid_eq = fid)['orientation_pos'][0] == node_visited:
        o_new = 'neg'
    else:
        print "Something went wrong."

    return o_new

def sample_particle_with_noise(GPSpoint,sigma_dist):
    e_new = random.gauss(GPSpoint.e, sigma_dist)
    n_new = random.gauss(GPSpoint.n, sigma_dist)
    s_new = random.gauss(GPSpoint.s, sigma_dist)
    b_new = random.gauss(GPSpoint.b, sigma_dist)
    
    return Point(e_new,n_new,s_new,b_new,GPSpoint.dt)

# In[28]:

def sample_particle_with_noise_network(particle,sigma_dist, street_network):
    """
    Sample new particle from a Gaussian distribution centered at particle with standard deviation sigma.
    """
    
    # Unpack street network
    camden_network = street_network[0]
    

    # Check whether there are any flow restrictions on the segment that the particle is on (e.g. one-wayness).
    street_orientation = camden_network.es.select(fid_eq = particle.fid)[0]['one_way']

    # Ensure that particle in agreement with street orientation:
    if not validate_particle_orientation(particle,street_network):
        print street_orientation
        print particle.o
        raise Exception("Input particle has incorrect orientation")

    # Sample position and bearing
    # ---------------------------
    # If street is one-way, displacement cannot be negative
    if True:
    #if street_orientation != '':
        while True:
            displacement = random.gauss(0, sigma_dist)
            if displacement >= 0:
                break
    # Else, any displacement is allowed.
    else:
        displacement = random.gauss(0, sigma_dist)

    # If displacement is negative
    if displacement < 0:
        # Change particle's orientation of travel for the move, move, and then reverse the orientation again
        # (unless ends up on a one-way street ==> then do not change the direction of travel from the original)
        if particle.o == 'pos':
            particle.o = 'neg'
            particle = move_particle(particle,abs(displacement), street_network)
            # If particle can change direction of travel, change it.
            if len(camden_network.es.select(fid_eq = particle.fid))>1:
                if particle.o == 'pos':
                    particle.o = 'neg'
                else:
                    particle.o = 'pos'
        elif particle.o == 'neg':
            particle.o = 'pos'
            particle = move_particle(particle,abs(displacement), street_network)
            # If particle can change direction of travel, change it.
            if len(camden_network.es.select(fid_eq = particle.fid))>1:
                if particle.o == 'pos':
                    particle.o = 'neg'
                else:
                    particle.o = 'pos'
    else:
        particle = move_particle(particle,displacement, street_network)

    # Sample speed
    # ------------
    # If street is one-way, speed cannot be negative

    if True:
    #street_orientation2 = camden_network.es.select(fid_eq = particle.fid)[0]['one_way']
    #if street_orientation2 != '':
        while True:
            speed = random.gauss(particle.s, sigma_dist)
            if speed >= 0:
                break
    # Else, any displacement is allowed.
    else:
        speed = random.gauss(particle.s, sigma_dist)

    # If speed is negative, change particle's direction of travel
    if speed < 0:
        particle.s = abs(speed)
        if particle.o == 'pos':
            particle.o = 'neg'
        elif particle.o == 'neg':
            particle.o = 'pos'
    else:
        particle.s = speed

    """
    # Before outputting, doucle check that particle's orientation is not in conflict with its street orientation.
    #street_orientation2 = camden_network.es.select(fid_eq = particle.fid)[0]['one_way']

    # Ensure that particle in agreement with street orientation:
    if not validate_particle_orientation(particle):
        print street_orientation2
        print particle.o
        raise Exception("Incorrect particle orientation")
    """
    return particle, displacement


# In[29]:

def validate_particle_orientation(particle,street_network):
    # Double check that particle's orientation is not in conflict with its street orientation.
    
    # Unpack street network
    camden_network = street_network[0]
    
    validated = True
    street_orientation = camden_network.es.select(fid_eq = particle.fid)[0]['one_way']

    # Ensure that particle in agreement with street orientation:
    if street_orientation!= '' and street_orientation != particle.o:
        validated = False

    return validated


# In[30]:

def weighted_choice(choices):
    """Like random.choice, but each element can have a different chance of
    being selected.

    choices can be any iterable containing iterables with two items each.
    Technically, they can have more than two items, the rest will just be
    ignored.  The first item is the thing being chosen, the second item is
    its weight.  The weights can be any numeric values, what matters is the
    relative differences between them.
    """
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for i, (c, w) in enumerate(choices):
       if upto + w > r:
          return c, i
       upto += w
    assert False, "Shouldn't get here"


# In[31]:

#get_ipython().magic(u'matplotlib inline')
def plot_particles(S,GPS_point,street_network,idx=0):
    """
    Plot particles and the corresponding GPS point. Set idx to -1 if you do not want to save figure.
    """
    
    # Unpack street network
    edge_polylines = street_network[2]
    
    # Plot background
    plt.figure()
    for segment in edge_polylines.values():
        coords = zip(*segment)        # zip + * unzips a list!
        plt.plot(coords[0],coords[1], 'grey')

    # Normalize weights for plotting
    weights = []
    for particle, weight in S:
        weights.append(weight)
    #print "Weights normalization"
    #weights_normalized = normalize_variable(weights)

    # Colormap settings
    norm = mpl.colors.Normalize(vmin=0, vmax=max(weights))
    cmap = cm.coolwarm
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Plot particles
    es = []
    ns = []
    es.append(GPS_point.e)
    ns.append(GPS_point.n)   
    for i, (particle, weight) in enumerate(S):
        e,n = get_particle_e_n(particle,street_network)
        plt.plot(e, n, 'o',markersize=6, color = m.to_rgba(weight),linewidth=0,markeredgecolor='none')  #weights_normalized[i]*10)
        es.append(e)
        ns.append(n)
    plt.plot(GPS_point.e, GPS_point.n, color ='lightgrey', marker = 'o', markersize = 12)
    plt.xlim(min(es)-50, max(es)+50)
    plt.ylim(min(ns)-50, max(ns)+50)
    if idx != -1:
        plt.savefig('results/iterations_sigma10/plot'+str(idx)+'_sigma10.png',dpi=400)


# In[32]:

def normalize_variable(values):
    """
    Normalize vales in list 'values' to 0-1 range.
    """

    mymin = np.min(values)
    mymax = np.max(values)

    if mymin==mymax:
        return mymin*np.ones((1,len(values)))[0]

    numerator = np.asarray(values,dtype=float)-mymin
    denominator = mymax-mymin

    return numerator/denominator


# In[33]:

def close_pass_filter(X, GPS_point, d=200):
    """
    For a given GPS point, returns the indices of vectors in X
    that are closer than a given distance d.
    """

    assert(type(X)==np.ndarray)
    assert(len(X.shape)==2)

    # np.where returns a tuple where the first element is the array of indices
    # that we are after
    return np.where(np.asarray([np.linalg.norm(x) for x in X-np.asarray([GPS_point.e,GPS_point.n])])<=d)[0]


# In[34]:

def is_point_on_segment(a,b,point, epsilon):
    """
    Check if point is on a line segment joining points a and b.
    """

    crossproduct = (point.n - a[1]) * (b[0] - a[0]) - (point.e - a[0]) * (b[1] - a[1])
    if abs(crossproduct) > epsilon : return False, distance_between_points(a, b)    # (or != 0 if using integers)

    dotproduct = (point.e - a[0]) * (b[0] - a[0]) + (point.n - a[1])*(b[1] - a[1])
    if dotproduct < 0 : return False, distance_between_points(a, b)

    squaredlengthba = (b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1])
    if dotproduct > squaredlengthba: return False, distance_between_points(a, b)

    distance_along_segment = min(distance_between_points(a, b), distance_between_points(a, [point.e,point.n])) # Make sure that particle does not venture outside the segment (due to epsilon tolerance).

    return True, distance_along_segment


# In[35]:

def is_point_moving_along_segment(a,b,segment_orientation,point,epsilon):
    """
    Check if point is moving along a segment by checking its location and bearing.
    """

    # Is point location in agreement with the segment?
    is_position_on_segment, distance = is_point_on_segment(a,b,point,5)

    # Is point bearing in agreement with the segment?
    # Option 1: One-way 'positive' street
    if segment_orientation == 'pos':
        diff_pos = get_angle_difference(point.b,get_segment_bearing(a,b))
        diff_neg = epsilon + 10    # make sure it is bigger than epsilon
    # Option 2: One-way 'negative' street
    elif segment_orientation == 'neg':
        diff_pos = epsilon + 10
        diff_neg = get_angle_difference(point.b,get_segment_bearing(b,a))
    # Option 3: Two-way street
    else:
        diff_pos = get_angle_difference(point.b,get_segment_bearing(a,b))
        diff_neg = get_angle_difference(point.b,get_segment_bearing(b,a))

    if is_position_on_segment:

        if diff_pos <= epsilon:
            return True, 'pos', distance
        if diff_neg <= epsilon:
            return True, 'neg', distance
    return False, 'pos', distance


# In[36]:

def get_segment_bearing(a,b):
    """
    Return bearing of a segment from point a to b.
    """
    return math.degrees(math.atan2(float(b[0]-a[0]), float(b[1]-a[1]))) % 360


# In[37]:

def get_angle_difference(angle1,angle2):
    """
    Return absolute difference between angles 1 and 2 (in degrees).
    """

    diff = angle1 - angle2    # a = targetA - sourceA
    diff = abs((diff + 180) % 360 - 180)

    return diff


def visualise_segment(segment_fid,street_network):
    
    # Unpack street network
    edge_polylines = street_network[2]
    
    segment = edge_polylines[segment_fid]
    for key,seg in edge_polylines.items():
        #plt.plot(seg[:,0],seg[:,1],colours[len(camden_network.es.select(fid_eq = key))])
        plt.plot(seg[:,0],seg[:,1],'grey')
    plt.plot(segment[:,0],segment[:,1],'green')
    plt.xlim(min(segment[:,0])-250, max(segment[:,0])+250)
    plt.ylim(min(segment[:,1])-250, max(segment[:,1])+250)
    


