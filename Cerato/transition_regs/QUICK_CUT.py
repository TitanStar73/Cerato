from transition_loader import transitions
TRANSITIONS = {
    "default": {
        "func": transitions.duck_rotate_breathe,
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
    "chain_linear_cut": {
        "func": transitions.linear_cut,
        "lowest_bpm": 240, #Acceptable bpms
        "highest_bpm": 2000000,
        "weight": 3, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"chain_linear_cut":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"chain_linear_cut":-1_000_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "fade_to_black_cut": {
        "func": transitions.fade_to_black_cut,
        "lowest_bpm": 140, #Acceptable bpms
        "highest_bpm": 400,
        "weight": 10, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"fade_to_black_cut":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"fade_to_black_cut":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "fade_to_white_cut": {
        "func": transitions.fade_to_white_cut,
        "lowest_bpm": 140, #Acceptable bpms
        "highest_bpm": 400,
        "weight": 10, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"fade_to_white_cut":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"fade_to_white_cut":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "my_neon_template": {
        "func": transitions.my_neon_template,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 50, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"my_neon_template":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"my_neon_template":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "neon_contour_switch_up": {
        "func": transitions.neon_contour_switch_up,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 500, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"neon_contour_switch_up":-100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"neon_contour_switch_up":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "white_cut": {
        "func": transitions.white_cut,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 300, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"white_cut":+300}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"white_cut":+300}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "bright_cut": {
        "func": transitions.bright_cut,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 300, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"bright_cut":+300}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"bright_cut":+300}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "bright_cut_breathe": {
        "func": transitions.bright_cut_breathe,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 300,
        "weight": 300, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"bright_cut_breathe":+300}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"bright_cut_breathe":+300}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "trip_bars_padding": {
        "func": transitions.trip_bars_padding,
        "lowest_bpm": 200, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 10_000, #Weight for random functions
        "beats": 4,
        "intro_priority": False,
        "boosts": {"trip_bars_padding":-10_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"trip_bars_padding":+10_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": True,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
       "trip_bars_padding_synth": {
        "func": transitions.trip_bars_padding_synth,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 100, #Weight for random functions
        "beats": 2,
        "intro_priority": False,
        "boosts": {"trip_bars_padding_synth":-0}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"trip_bars_padding_synth":+0}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": True,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "adaptive_bars_lean1": {
        "func": transitions.adaptive_bars_lean,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 100, #Weight for random functions
        "beats": 2,
        "intro_priority": False,
        "boosts": {"adaptive_bars_lean":-0}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"adaptive_bars_lean":+0}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": True,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "adaptive_bars_lean2": {
        "func": transitions.adaptive_bars_lean,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 100, #Weight for random functions
        "beats": 3,
        "intro_priority": False,
        "boosts": {"adaptive_bars_lean":-0}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"adaptive_bars_lean":+0}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": True,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "adaptive_bars_lean3": {
        "func": transitions.adaptive_bars_lean,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 100, #Weight for random functions
        "beats": 4,
        "intro_priority": False,
        "boosts": {"adaptive_bars_lean":-0}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"adaptive_bars_lean":+0}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": True,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "quad_duo_silhoutte_cut_smart": {
        "func": transitions.quad_duo_silhoutte_cut_smart,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 2500, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"quad_duo_silhoutte_cut_smart":+1_00_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"tri_mono_silhoutte_cut_smart":-1_00_000, "quad_duo_silhoutte_cut_smart":-1_00_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "tri_mono_silhoutte_cut_smart": {
        "func": transitions.tri_mono_silhoutte_cut_smart,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 2500, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"tri_mono_silhoutte_cut_smart":+1_00_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"tri_mono_silhoutte_cut_smart":-1_00_000, "quad_duo_silhoutte_cut_smart":-1_00_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "partial_image_quick_cut": {
        "func": transitions.partial_image_quick_cut,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 2500, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"partial_image_quick_cut":+1_00_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"partial_image_quick_cut":-1_00_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "zoom_out_ns_block": {
        "func": transitions.zoom_out_ns_block,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 300, #Weight for random functions
        "beats": 2,
        "intro_priority": False,
        "boosts": {"zoom_out_ns_block":+0}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"zoom_out_ns_block":+0}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": True,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "zoom_out_silohoutte_block_white": {
        "func": transitions.zoom_out_silohoutte_block_white,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 300, #Weight for random functions
        "beats": 2,
        "intro_priority": False,
        "boosts": {"zoom_out_silohoutte_block_white":+0}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"zoom_out_silohoutte_block_white":+0}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": True, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": True,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "zoom_out_silohoutte_block_black": {
        "func": transitions.zoom_out_silohoutte_block_black,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 300, #Weight for random functions
        "beats": 2,
        "intro_priority": False,
        "boosts": {"zoom_out_silohoutte_block_black":+0}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"zoom_out_silohoutte_block_black":+0}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": True, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": True,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "image_quick_cut_bright2": {
        "func": transitions.image_quick_cut_bright2,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 300, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"image_quick_cut_bright2":+1_00_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"image_quick_cut_bright2":-4_00_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "image_quick_cut_bright": {
        "func": transitions.image_quick_cut_bright,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 300, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"image_quick_cut_bright":+1_00_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"image_quick_cut_bright":-4_00_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    


}