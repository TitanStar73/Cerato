from transition_loader import transitions

TRANSITIONS = {
    "default": {
        "func": transitions.linear_cut,
        "lowest_bpm": 1, #Acceptable bpms
        "highest_bpm": 2000000,
        "weight": 1, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {}, #Dictionary will names of transitions that will boost its weight if done in the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },
    "cloud_bars": {
        "func": transitions.cloud_bars,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 90,
        "weight": 100, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"cloud_bars":-100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "cloud_bars_spawn_in": {
        "func": transitions.cloud_bars_spawn_in,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 150,
        "weight": 100, #Weight for random functions
        "beats": 2,
        "intro_priority": False,
        "boosts": {"cloud_bars_spawn_in":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"cloud_bars_spawn_in":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "cloud_bars_spawn_in_vert": {
        "func": transitions.cloud_bars_spawn_in_vert,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 150,
        "weight": 100, #Weight for random functions
        "beats": 2,
        "intro_priority": False,
        "boosts": {"cloud_bars_spawn_in_vert":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"cloud_bars_spawn_in_vert":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "cloud_bars_slide_in": {
        "func": transitions.cloud_bars_slide_in,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 180,
        "weight": 100, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"cloud_bars_slide_in":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"cloud_bars_slide_in":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "cloud_bars_slide_in2": {
        "func": transitions.cloud_bars_slide_in,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 180,
        "weight": 100, #Weight for random functions
        "beats": 2,
        "intro_priority": False,
        "boosts": {"cloud_bars_slide_in2":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"cloud_bars_slide_in2":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "cloud_pulse_slide": {
        "func": transitions.cloud_pulse_slide,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 180,
        "weight": 100, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"cloud_pulse_slide":-100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"cloud_pulse_slide":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "black_slides": {
        "func": transitions.black_slides,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 350, #Weight for random functions
        "beats": 2,
        "intro_priority": True,
        "boosts": {"black_slides":-100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"black_slides":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":True, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "ripple_cut": {
        "func": transitions.ripple_cut,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 350, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"ripple_cut":-100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"ripple_cut":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "glitch_cut": {
        "func": transitions.glitch_cut,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 200, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"glitch_cut":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"glitch_cut":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":True, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "glitch_cut2": {
        "func": transitions.glitch_cut2,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 350, #Weight for random functions
        "beats": 1,
        "intro_priority": True,
        "boosts": {"glitch_cut2":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"glitch_cut2":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":True, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "slide_shake_violent": {
        "func": transitions.slide_shake_violent,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 100, #Weight for random functions
        "beats": 2,
        "intro_priority": False,
        "boosts": {"slide_shake_violent":+1_00_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"slide_shake_violent":-5_00_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    


}

