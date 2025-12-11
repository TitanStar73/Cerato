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
    "camera_shake_cut": {
        "func": transitions.camera_shake_cut,
        "lowest_bpm": 30, #Acceptable bpms
        "highest_bpm": 180,
        "weight": 100, #Weight for random functions
        "beats": 1,
        "intro_priority": False,
        "boosts": {"camera_shake_cut":+100_000}, #Dictionary with all names of transitions that will boost its weight (both negatively or positively, less than 0 = no chance)
        "boosts_long_term": {"camera_shake_cut":-500_000}, #Dictionary will names of transitions that will boost its weight if done after the past 3 beats
        "requires_rb": False, #Gives different block_progress btw, more accurate if requires_rb = False -> (f1 bp, f2 bp) | If true u gotta just take f1 bp = 1, f2 bp = 0
        "prefers_video": False, #NOT A GUARENTEE of video as either, but may provide IF ENOUGH
        "extend_block_progress": False,
        "requires_rb_in":False, #Overrides requires_rb normally, just remove requires_rb if you use this. Using this will get you a (f1,f2) block progress
        "requires_rb_out":False
    },    

}