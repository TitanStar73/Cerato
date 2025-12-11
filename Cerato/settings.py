UHD = False
VIDEO_SIZE = (3840,3840) if UHD else (1080,1080) #w x h
FPS = 60 if UHD else 30

ideal_duration_buffer_margin = 1.25 #Duration margin in terms of ideal duration listed in the transition registry


import getpass
USERNAME = str(getpass.getuser())
PROJECT_FOLDER = f"C:/Users/{USERNAME}/Downloads" #Parent folder for all projects

RB_MODEL = 'u2net_human_seg' #or 'isnet-general-use' or 'u2net_human_seg'
#THIS SETTING WILL CAUSE NO TRUE IMAGE/VIDEO TIME
TOTAL_TRANSITION_TIME = False #Only transitions, useful if using filters EVERYWHERE, prolly better to just change block_splits tho
block_splits = (0.5,0.5) #Block splits for In-transition, actual thing, out-transition

MAX_MEDIA_USAGE = 0.0 #Max media usage in seconds in output file (videos will have fps boosted to meet this requirement) | 0.0 disables this

#Control chronology:
top_files = [] #Always selected among peers
#P{x}.mp4 -> Higher number selected first
#N{x}.mp4 -> Higher number selected last

#Text settings
TEXT_FADE_LEN = 0.4 #Fade length in seconds
TEXT_FADE_PROG = lambda x:x #Fade progress function
TEXT_FAST_SPEAK = False #For fast speech (max 1 in each layer)
TEXT_ENABLE = (True, True, True) #Before fade, during, after fade

#Linear sequence settings
ENABLE_VIDEO_SNAPPING = True #If true, video blocks will expand to try and maintain their true timing (and 'snap' to beats)
ENABLE_HOOK_INJECTION = False #If using a hook video, and this is true, it will create temporary files and try to integrate the hook video into the linear-seq
LS_REBALANCER = True #Works with video snapping. Attempts to 'rebalance' indiviual video freq. BREAKS if all videos do not have a removed background

#Debug settings
DEBUG_MODE = False
DISABLE_EDITS = False #Disable the editor (LUTs and GUI-editor)
ALWAYS_MAKE_NEW_LS = True #Always make a new linear sequence
ALWAYS_ASK_FOR_SONG = False
ENABLE_TRANSITION_PROMPTS = True #If true, after transitions prompt you with suggestions, they may wait for addtional input before continuing allowing you time to implement the changes if you want. (Some may still ask for non-optional input)
