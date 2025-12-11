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
    "pulse_cut": {
        "func": transitions.pulse_cut,
        "lowest_bpm": 80, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 100, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"pulse_cut":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"pulse_cut":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "large_pulse_cut": {
        "func": transitions.large_pulse_cut,
        "lowest_bpm": 35, #Acceptable bpms
        "highest_bpm": 250,
        "weight": 100, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"large_pulse_cut":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"large_pulse_cut":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "zoom_through_cut": {
        "func": transitions.zoom_through_cut,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 200,
        "weight": 100, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"zoom_through_cut":-100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "slide_cut_top": {
        "func": transitions.slide_cut_top,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 25, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"slide_cut_bottom":+100_000, "slide_cut_top":-1_000_000, }, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"slide_cut_top":-500_000,"slide_cut_bottom":-500_000, "slide_cut_left":+300_000,"slide_cut_right":+300_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "slide_cut_bottom": {
        "func": transitions.slide_cut_bottom,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 25, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"slide_cut_top":+100_000,"slide_cut_bottom":-1_000_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"slide_cut_top":-500_000,"slide_cut_bottom":-500_000, "slide_cut_left":+300_000,"slide_cut_right":+300_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    }, 
    "slide_cut_right": {
        "func": transitions.slide_cut_right,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 25, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"slide_cut_left":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"slide_cut_top":-500_000,"slide_cut_bottom":-500_000, "slide_cut_left":-500_000,"slide_cut_right":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "slide_cut_left": {
        "func": transitions.slide_cut_left,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 25, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"slide_cut_right":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"slide_cut_top":-500_000,"slide_cut_bottom":-500_000, "slide_cut_left":-500_000,"slide_cut_right":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    }, 
     "slide_cut_smart": {
        "func": transitions.slide_cut_smart,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 200, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"slide_cut_smart":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"slide_cut_smart":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    


    "slide_cut_top_bright": {
        "func": transitions.slide_cut_top_bright,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 25, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"slide_cut_bottom_bright":+100_000, "slide_cut_top_bright":-1_000_000, }, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"slide_cut_top_bright":-500_000,"slide_cut_bottom_bright":-500_000, "slide_cut_left_bright":+300_000,"slide_cut_right_bright":+300_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "slide_cut_bottom_bright": {
        "func": transitions.slide_cut_bottom_bright,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 25, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"slide_cut_top_bright":+100_000,"slide_cut_bottom_bright":-1_000_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"slide_cut_top_bright":-500_000,"slide_cut_bottom_bright":-500_000, "slide_cut_left_bright":+300_000,"slide_cut_right_bright":+300_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    }, 
    "slide_cut_right_bright": {
        "func": transitions.slide_cut_right_bright,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 25, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"slide_cut_left_bright":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"slide_cut_top_bright":-500_000,"slide_cut_bottom_bright":-500_000, "slide_cut_left_bright":-500_000,"slide_cut_right_bright":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "slide_cut_left_bright": {
        "func": transitions.slide_cut_left_bright,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 25, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"slide_cut_right_bright":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"slide_cut_top_bright":-500_000,"slide_cut_bottom_bright":-500_000, "slide_cut_left_bright":-500_000,"slide_cut_right_bright":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    }, 
     "slide_cut_smart_bright": {
        "func": transitions.slide_cut_smart_bright,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 200, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"slide_cut_smart_bright":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"slide_cut_smart_bright":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "radio_blur_cut": {
        "func": transitions.radio_blur_cut,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 350, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"radio_blur_cut":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"radio_blur_cut":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
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
    "slide_shake_brr": {
        "func": transitions.slide_shake_brr,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 350, #Weight for random functions
        "beats": 2,
        "intro_priority": False,
        "boosts": {"slide_shake_brr":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"slide_shake_brr":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "shake_brr": {
        "func": transitions.shake_brr,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 450, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"shake_brr":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"shake_brr":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "slide_fade_zoom": {
        "func": transitions.slide_fade_zoom,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 100, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"adaptive_bars_lean":-0}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"adaptive_bars_lean":+0}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "crossfade_cut": {
        "func": transitions.crossfade_cut,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 10, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"crossfade_cut":-0}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"crossfade_cut":+0}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "slide_shake_squeeze": {
        "func": transitions.slide_shake_squeeze,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 100, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"slide_shake_squeeze":-0}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"slide_shake_squeeze":+0}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "slide_shake_violent": {
        "func": transitions.slide_shakex_smart,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 400, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"slide_shake_violent":-1_00_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"slide_shake_violent":-5_00_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "skin_effect_cut": {
        "func": transitions.skin_effect_cut,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 300, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"skin_effect_cut":+0}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"skin_effect_cut":+0}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "zoom_pan_in": {
        "func": transitions.zoom_pan_in,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 1000, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"zoom_pan_in":+1_00_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"zoom_pan_in":-4_00_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":True, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    

}