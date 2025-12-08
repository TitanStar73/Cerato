#Basically just some progress functions, mostly mapping 0 -> 1 to 0 -> 1; used for time-muxing and that sort

import math
import numpy as np
import random

def linear(x):
    return x

def pentic(x):
    return x**5

def ease_in_out_sine(x):
    return -(math.cos(math.pi * x) - 1) / 2

def sin(x):
    return math.sin(x * math.pi * 2)

def tepee(x):
    if x < .5:
        return 2 * x
    else:
        return 2 - 2*x

def square(x):
	return x**2

def deca(x):
	return x**10

def quick(x):
    return x**3.5

def cubic(x):
    return x**3

def sqrt(x):
    return x**0.5

def inf_to_unit(x):
    """ -inf to inf -> -1 to 1"""
    return math.tanh(1.5 * x)

def parametric_curve(t, A, B, m, n, phi_x, phi_y):
    """General parametric curve function. -> Not limited to [-1,1]"""
    X = np.sin(2 * np.pi * (m * t + phi_x)) + A * t * (1 - t)
    Y = np.cos(2 * np.pi * (n * t + phi_y)) + B * t * (1 - t)
    return X, Y

PARAMTERIC_SCALE_REGISTRY = {}
def scaled_curve(t, A, B, m, n, phi_x, phi_y, samples=1000):
    """Compute curve then scale so it's within [-1,1] for both x and y."""
    # Sample curve densely to estimate maxima/minima
    key = f"{round(float(A), 2)}_{round(float(B), 2)}_{round(float(m),2)}_{round(float(n),2)}_{round(float(phi_x), 2)}_{round(float(phi_y),2)}"
    
    global PARAMTERIC_SCALE_REGISTRY
    if key in PARAMTERIC_SCALE_REGISTRY:
        return PARAMTERIC_SCALE_REGISTRY[key]
    
    ts = np.linspace(0, 1, samples)
    X, Y = parametric_curve(ts, A, B, m, n, phi_x, phi_y)
    
    max_abs_x = np.max(np.abs(X))
    max_abs_y = np.max(np.abs(Y))
    
    # Scale factor to fit within [-1,1]
    scale = 1.0 / max(max_abs_x, max_abs_y, 1.0)
    
    PARAMTERIC_SCALE_REGISTRY[key] = scale
    return scale

def parametric_curve_from_seed_scaled(t, seed, bias, bias_strength):
    """Gives (x,y) points -> ([-1,1],[-1,1])"""
    
    #m,n -> +- 2.5 and atleast +-.5, also 
    #A,B -> +- 2
    #p,q 0 to 1
    bias = [bias[0] * -math.sin(t * math.pi), bias[1] * math.sin(t * math.pi)] #Y is on the wrong side


    random.seed(seed)
    m = (-1 if random.randint(0,1) == 0 else 1) * random.randint(500, 2500)/1000
    n = (-1 if random.randint(0,1) == 0 else 1) * random.randint(500, 2500)/1000
     
    A = random.randint(-2000, 2000)/1000
    B = random.randint(-2000, 2000)/1000
    
    p = random.randint(0, 1000)/1000
    q = random.randint(0, 1000)/1000

    X,Y = parametric_curve(t, A, B, m, n, phi_x = p, phi_y=q)
    scale = float(scaled_curve(t, A, B, m, n, phi_x = p, phi_y=q))

    X,Y = bias[0] * bias_strength + (1-bias_strength) * X, bias[1] * bias_strength + (1-bias_strength) * Y 
    
    return  (float(X) * scale, float(Y) * scale)
    

def elastic_out(x):
    c4 = (2 * math.pi) / 3
    if x == 0 or x == 1:
       return x
    return pow(2, -10 * x) * math.sin((x * 10 - 0.75) * c4) + 1

def bouncy(x):
    x2 = (x * (0.3*math.pi - 0.13)) + 0.13
    x2 = min(max(0,x2),1)
    return min(1, 2**(-3.9 * x2) * math.sin(10*x2) * 1.5)

def explosion(x):
    return x**.5

def sharp_curve_off(x):
    """This will goes from 1 to 0"""
    if x < 0.5:
        return 1
    else:
        return (1 - 2*(x-0.5))**.07
     
def parabolic_tepee(x):
    return -4*x*x + 4*x

def asymmetric_parabolic_tepee(x, center = 0.7, l =3.1):
    if center == 0:
        raise Exception("Center is 0")
    if x < 0 or x > 1:
        raise Exception(f"X is {x}")


    a = l * center
    b = l * (1 - center)

    return ((x**a) * ((1-x)**b))/((center**a) * ((1-center)**b))


def smooth_stop_in_between(x):
    return x+0.18*math.sin(2*math.pi*x)

def smooth_stop_in_between_adv(x, center_y_val):
    prev_y_val = .5
    if x <= .5:
        return smooth_stop_in_between(x) * (center_y_val/.5)
    return ((smooth_stop_in_between(x) - prev_y_val)/.5) * (1 - center_y_val)

def smooth_stop_in_between_extra_adv(x, center_y_val, pause_dist):
    a,b = .5 - (pause_dist/2), .5 + (pause_dist/2)

    if x < a:
        return smooth_stop_in_between_adv((x/a) * .5, center_y_val)
    elif x > b:
        return center_y_val + smooth_stop_in_between_adv((((x-b)/(1-b)) * .5) + .5, center_y_val)
    
    return smooth_stop_in_between_adv(.5, center_y_val)

def quick_stop_in_between(x):
    if x <=.5:
        return (0.5**0.5)*(x**0.5)
    return (((2*(x-.5))**3.5) + 1)/2

def damped_jitter(
    p,
    jitter_x=0.7, jitter_y=0.7,        # [0,1] relative amplitude
    cycles_x=3.0, cycles_y=2.5,       # number of full oscillation-cycles over p∈[0,1]
    seed=None,                        # optional int for deterministic jitter
    warp=0.6,                         # <1 => faster early, >1 => slower early
    amp_decay=1.2,                    # exponent for amplitude envelope (higher => faster amp drop)
    freq_spread=0.12                  # random per-axis frequency variation fraction
):
    """
    Return (x,y) in [-1,1] for progress p in [0,1].

    Behavior:
      - Each axis is an independent damped sinusoid.
      - Amplitude envelope ~ jitter * (1-p)**amp_decay (large at start, small at end).
      - Time warp tau = p**warp makes motion faster early when warp<1.
      - Small random phase and frequency variation per axis controlled by `seed`.
    """
    # clamp progress
    if p <= 0.0:
        return (0.0, 0.0)
    if p >= 1.0:
        return (0.0, 0.0)

    # deterministic RNG if seed given
    rnd = random.Random(seed)

    def axis(jitter, cycles, phase_seed_offset):
        if jitter <= 0 or cycles == 0:
            return 0.0

        # derive per-axis pseudo-random constants but stable for same seed
        phase0 = rnd.uniform(0.0, 2.0 * math.pi) + phase_seed_offset
        freq_var = 1.0 + rnd.uniform(-freq_spread, freq_spread)

        # non-linear time mapping to bias speed toward start (warp < 1 => faster early)
        tau = p ** warp

        phi = 2.0 * math.pi * cycles * tau * freq_var + phase0

        amp = jitter * ((1.0 - p) ** amp_decay)

        damp = math.exp(-2.5 * p)
        val = amp * damp * math.sin(phi)

        return max(-1.0, min(1.0, val))

    # produce independent x,y (phase offsets keep them different even with same seed)
    x = axis(jitter_x, cycles_x, phase_seed_offset=0.0)
    y = axis(jitter_y, cycles_y, phase_seed_offset=37.0)

    return (x, y)

def quick_dip_rebound(x):
    return 10*(x**5) - 11*(x**4) + 1



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Parameters
    center_y_val = 0.6
    pause_dist = 0.3

    # Generate values
    x_vals = np.linspace(0, 1, 500)
    y_vals = [smooth_stop_in_between_extra_adv(x, center_y_val, pause_dist) for x in x_vals]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label='smooth_stop_in_between_extra_adv', linewidth=2)
    plt.title(f'smooth_stop_in_between_extra_adv(center_y_val={center_y_val}, pause_dist={pause_dist})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()
