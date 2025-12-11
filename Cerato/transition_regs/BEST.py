from transition_loader import transitions

TRANSITIONS = {
    "default": {
        "func": transitions.fade_to_black_cut,
        "lowest_bpm": 140, #Acceptable bpms
        "highest_bpm": 400,
        "weight": 3, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"default":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"default":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
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
        "weight": 500, #Weight for random functions
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
    "slide_shake_violent": {
        "func": transitions.slide_shakex_smart,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 700, #Weight for random functions
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
    "skin_effect_cut": {
        "func": transitions.skin_effect_cut,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 40, #Weight for random functions
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