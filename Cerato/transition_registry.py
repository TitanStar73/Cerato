import transitions

SLIDE_BEATS = 4 #Beats for all slide transitions
SLIDE_WEIGHTS = .5 #6 slide transitions btw

TRANSITIONS = {
    "linear_cut": {
        "func": transitions.linear_cut,
        "lowest_bpm": 1, #Acceptable bpms
        "highest_bpm": 20000,
        "weight": 100, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {}, #Dictionary will names of transitions that will boost its weight if done in the past 10 beats
        "requires_rb": False,
        "prefers_video": False #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
    },
    "pulse_cut": {
        "func": transitions.pulse_cut,
        "lowest_bpm": 1, #Acceptable bpms
        "highest_bpm": 20000,
        "weight": 10, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {}, #Dictionary will names of transitions that will boost its weight if done in the past 10 beats
        "requires_rb": False,
        "prefers_video": False #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
    },
    "large_pulse_cut": {
        "func": transitions.large_pulse_cut,
        "lowest_bpm": 1, #Acceptable bpms
        "highest_bpm": 20000,
        "weight": 3, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {}, #Dictionary will names of transitions that will boost its weight if done in the past 10 beats
        "requires_rb": False,
        "prefers_video": False #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
    },
    "bouncy_stripes_cut": {
        "func": transitions.bouncy_stripes_cut,
        "lowest_bpm": 1, #Acceptable bpms
        "highest_bpm": 20000,
        "weight": 0, #Weight for random functions
        "beats": 12,
        "intro_priority": False,
        "boosts": {}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {'bouncy_stripes_cut': -10}, #Dictionary will names of transitions that will boost its weight if done in the past 10 beats
        "requires_rb":False
    },
    "title_card": {
        "func": transitions.title_card,
        "lowest_bpm": 1, #Acceptable bpms
        "highest_bpm": 20000,
        "weight": -1, #Weight for random functions -> -1 = 0 for intro_priority -> (w) * 1e9
        "beats": 5,
        "intro_priority": False,
        "boosts": {}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {}, #Dictionary will names of transitions that will boost its weight if done in the past 10 beats
        "requires_rb": False
    },
    "cloud_pulse": {
        "func": transitions.linear_cut,
        "lowest_bpm": 1, #Acceptable bpms
        "highest_bpm": 20000,
        "weight": 1000, #Weight for random functions -> -1 = 0 for intro_priority -> (w) * 1e9
        "beats": 1,
        "intro_priority": False,
        "boosts": {}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {'cloud_pulse':-10}, #Dictionary will names of transitions that will boost its weight if done in the past 10 beats
        "requires_rb": True
    },  

    "silhouette_cut_white": {
        "func": transitions.silhouette_cut_white,
        "lowest_bpm": 1, #Acceptable bpms
        "highest_bpm": 20000,
        "weight": 0, #Weight for random functions
        "beats": 2,
        "intro_priority": False,
        "boosts": {}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {}, #Dictionary will names of transitions that will boost its weight if done in the past 10 beats
        "requires_rb": True,
        "prefers_video": False #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
    },
    "slide_shake_squeeze": {
        "func": transitions.duck_rotate_breathe,
        "lowest_bpm": 1, #Acceptable bpms
        "highest_bpm": 20000,
        "weight": 1000000, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {}, #Dictionary will names of transitions that will boost its weight if done in the past 10 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },

    
}


NEW_TRANSITIONS = {
    
}

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

}

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


BCQ_BEATS = 5
import transitions
TRANSITIONS = {
    "default": {
        "func": transitions.radio_blur_cut,
        "lowest_bpm": 1, #Acceptable bpms
        "highest_bpm": 400,
        "weight": 3, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
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
        "weight": 300, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"zoom_pan_in":+1_00_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"zoom_pan_in":-1_00_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    
    "zoom_pan_in_non_chalant": {
        "func": transitions.zoom_pan_in_non_chalant,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 3000, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"zoom_pan_in":+1_00_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"zoom_pan_in":-1_00_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    }, 
    "bounce_zoom": {
        "func": transitions.zoom_pan_in_bounce,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 3000, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"bounce_zoom":+1_00_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"bounce_zoom":-1_00_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    


    "bqc1": {
        "func": transitions.bqc1,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 1_500, #Weight for random functions
        "beats": BCQ_BEATS,
        "intro_priority": False,
        "boosts": {"bqc1":-1_00_000, "bqc2":-1_00_000, "bqc3":-1_00_000, "bqc4":-1_00_000, "bqc5":-1_00_000, }, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"bqc1":-1_00_000, "bqc2":-1_00_000, "bqc3":-1_00_000, "bqc4":-1_00_000, "bqc5":-1_00_000, }, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },    
    "bqc2": {
        "func": transitions.bqc2,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 1_500, #Weight for random functions
        "beats": BCQ_BEATS,
        "intro_priority": False,
        "boosts": {"bqc1":-1_00_000, "bqc2":-1_00_000, "bqc3":-1_00_000, "bqc4":-1_00_000, "bqc5":-1_00_000, }, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"bqc1":-1_00_000, "bqc2":-1_00_000, "bqc3":-1_00_000, "bqc4":-1_00_000, "bqc5":-1_00_000, }, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },  
    "bqc3": {
        "func": transitions.bqc3,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 2_500, #Weight for random functions
        "beats": BCQ_BEATS,
        "intro_priority": False,
        "boosts": {"bqc1":-1_00_000, "bqc2":-1_00_000, "bqc3":-1_00_000, "bqc4":-1_00_000, "bqc5":-1_00_000, }, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"bqc1":-1_00_000, "bqc2":-1_00_000, "bqc3":-1_00_000, "bqc4":-1_00_000, "bqc5":-1_00_000, }, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },  
    "bqc4": {
        "func": transitions.bqc4,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 2_500, #Weight for random functions
        "beats": BCQ_BEATS,
        "intro_priority": False,
        "boosts": {"bqc1":-1_00_000, "bqc2":-1_00_000, "bqc3":-1_00_000, "bqc4":-1_00_000, "bqc5":-1_00_000, }, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"bqc1":-1_00_000, "bqc2":-1_00_000, "bqc3":-1_00_000, "bqc4":-1_00_000, "bqc5":-1_00_000, }, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },  
    "bqc5": {
        "func": transitions.bqc5,
        "lowest_bpm": 100, #Acceptable bpms
        "highest_bpm": 600,
        "weight": 8_000, #Weight for random functions
        "beats": BCQ_BEATS,
        "intro_priority": False,
        "boosts": {"bqc1":-1_00_000, "bqc2":-1_00_000, "bqc3":-1_00_000, "bqc4":-1_00_000, "bqc5":-1_00_000, }, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"bqc1":-1_00_000, "bqc2":-1_00_000, "bqc3":-1_00_000, "bqc4":-1_00_000, "bqc5":-1_00_000, }, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":True
    },  

}
#keys.extend([f"bqc{x}" for x in range(1,6)])


keys = ["shake_brr","default","white_cut", "zoom_pan_in","zoom_pan_in_non_chalant", "silocut", "lensblur"] #Add silocut
weights = [2500,1,600,300,900,3_000, 3_000]
funcs = [transitions.shake_brr,transitions.shake_brr,transitions.white_cut, transitions.zoom_pan_in,transitions.zoom_pan_in_non_chalant, transitions.silhouette_cut_white, transitions.lens_blur_cut]
chain = [True,False,True,False,True, True, False]

TEMPLATE = {
        "func": transitions.zoom_pan_in_non_chalant,
        "lowest_bpm": 1, #Acceptable bpms
        "highest_bpm": 2000,
        "weight": 3000, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": dict(), #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": dict(), #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    }

TRANSITIONS = {key:dict(TEMPLATE) for key in keys}

for i,key in enumerate(keys):
    TRANSITIONS[key]["weight"] = weights[i]
    TRANSITIONS[key]["func"] = funcs[i]
    if chain[i]:
        TRANSITIONS[key]["boosts"] = dict(TRANSITIONS[key].get("boosts", {}))
        TRANSITIONS[key]["boosts_long_term"] = dict(TRANSITIONS[key].get("boosts_long_term", {}))
        TRANSITIONS[key]["boosts"][key] = 100_000
        TRANSITIONS[key]["boosts_long_term"][key] = -400_000
        
TRANSITIONS['silocut']["requires_rb_out"] = True