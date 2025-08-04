import json
import csv
import re
from collections import defaultdict
from typing import List, Dict, Tuple
import random

class SemanticHashtagConnector:
    def __init__(self):
        self.artistic_analysis = []
        self.hashtags = []
        self.quote_to_images = {}
        
        # Define semantic themes and their associated keywords
        self.theme_keywords = {
            'death_mortality': ['death', 'dead', 'mortality', 'ghost', 'haunted', 'cemetery', 'funeral', 'grave'],
            'technology_digital': ['computer', 'digital', 'pixel', 'screen', 'code', 'program', 'system', 'update', 'error', 'glitch', 'internet', 'cloud'],
            'control_power': ['control', 'power', 'authority', 'state', 'government', 'crown', 'king', 'queen', 'sovereignty'],
            'war_conflict': ['war', 'battle', 'fight', 'weapon', 'sword', 'violence', 'conflict', 'resistance'],
            'love_desire': ['love', 'desire', 'romance', 'seduction', 'passion', 'heart', 'relationship'],
            'time_eternity': ['time', 'clock', 'eternal', 'infinity', 'circle', 'recurrence', 'moment'],
            'nature_organic': ['nature', 'earth', 'sky', 'stars', 'moon', 'sun', 'tree', 'flower', 'animal'],
            'human_body': ['body', 'skin', 'face', 'head', 'hand', 'eye', 'human', 'person'],
            'memory_secret': ['memory', 'secret', 'hidden', 'forgotten', 'past', 'history', 'archive'],
            'chaos_insanity': ['chaos', 'madness', 'insanity', 'confusion', 'disorder', 'random'],
            'creation_art': ['create', 'art', 'drawing', 'painting', 'design', 'craft', 'build'],
            'fear_dread': ['fear', 'dread', 'horror', 'terror', 'anxiety', 'panic', 'nightmare'],
            'hope_faith': ['hope', 'faith', 'belief', 'heaven', 'divine', 'god', 'spiritual'],
            'reality_illusion': ['reality', 'illusion', 'dream', 'fantasy', 'simulation', 'virtual'],
            'isolation_alienation': ['alone', 'isolated', 'alienation', 'lonely', 'separate', 'distant'],
            'routine_daily': ['routine', 'daily', 'everyday', 'ordinary', 'normal', 'regular'],
            'absurdity_meaningless': ['absurd', 'meaningless', 'nonsense', 'random', 'strange', 'weird'],
            'transformation_change': ['change', 'transform', 'evolve', 'become', 'shift', 'alter'],
            'boundary_limit': ['boundary', 'limit', 'border', 'edge', 'end', 'point', 'beyond'],
            'communication_text': ['text', 'word', 'letter', 'message', 'communication', 'writing', 'document']
        }
        
        # Define quote-specific keywords for better matching
        self.quote_specific_keywords = {
            'Sometimes_dead_is_better': ['death', 'cemetery', 'grave', 'dark', 'mysterious'],
            'Silence_is_aform_of_consent': ['silence', 'quiet', 'consent', 'agreement', 'passive', 'compliance', 'surrender', 'submission', 'obedience', 'acquiescence', 'yielding', 'capitulation'],
            'Beneath_the_skin_we\'re_already_ghosts': ['skin', 'ghost', 'body', 'transparent', 'ethereal'],
            'God_is_acircle_whose_center_is_everywhere': ['circle', 'center', 'divine', 'infinite', 'geometric'],
            'Everything_not_saved_will_be_lost': ['save', 'lost', 'preserve', 'memory', 'digital'],
            'Hell_is_ateenage_girl': ['teenage', 'girl', 'youth', 'hell', 'dramatic'],
            'Is_it_still_raining_I_hadn\'t_noticed': ['rain', 'weather', 'water', 'unaware', 'distracted'],
            'We_were_never_in_control': ['control', 'chaos', 'powerless', 'random', 'unpredictable'],
            'I_am_become_death_destroyer_of_worlds': ['death', 'destruction', 'power', 'atomic', 'catastrophe'],
            'This_is_water': ['water', 'liquid', 'flow', 'awareness', 'simple', 'ocean', 'river', 'drop', 'wave', 'sea', 'rain', 'stream', 'pond', 'lake', 'fluid', 'moisture', 'wet', 'drip', 'splash'],
            'Time_is_a_flat_circle': ['time', 'circle', 'flat', 'repetition', 'eternal', 'clock', 'hour', 'minute', 'second', 'moment', 'period', 'duration', 'cycle', 'loop', 'return', 'repeat', 'circular', 'round', 'spherical'],
            'Icould_eat_apeach_for_hours': ['peach', 'fruit', 'seduction', 'desire', 'sensual'],
            'These_violent_delights_have_violent_ends': ['violence', 'delight', 'conflict', 'drama', 'tragedy'],
            'Icould_be_bound_in_anutshell': ['nutshell', 'small', 'confined', 'infinite', 'contained'],
            'Even_the_stars_are_wrong_here': ['stars', 'sky', 'wrong', 'alien', 'cosmic'],
            'Its_afine_day_for_science': ['science', 'experiment', 'laboratory', 'discovery', 'knowledge'],
            'They_made_us_for_their_war': ['war', 'creation', 'purpose', 'conflict', 'manufactured'],
            'My_milkshake_brings_all_the_boys': ['milkshake', 'drink', 'attraction', 'desire', 'youth'],
            'Love_is_the_drug_and_iwont_give_up': ['love', 'drug', 'addiction', 'obsession', 'chemical'],
            'Death_is_the_road_to_awe': ['death', 'awe', 'transcendence', 'spiritual', 'mystical'],
            'I_am_tired_of_earth_these_people': ['earth', 'people', 'tired', 'alienation', 'exhaustion'],
            'The_motel_room_was_bleeding_light': ['motel', 'room', 'light', 'bleeding', 'surreal'],
            'Kill_me_softly_with_science': ['science', 'kill', 'soft', 'gentle', 'technology', 'laboratory', 'experiment', 'research', 'chemistry', 'biology', 'physics', 'microscope', 'test', 'analysis', 'discovery', 'innovation', 'invention', 'method', 'procedure', 'technique'],
            'We_all_go_alittle_mad_sometimes': ['madness', 'insanity', 'mental', 'psychology', 'human'],
            'This_is_not_ameeting_this_is_ahunger': ['meeting', 'hunger', 'desire', 'need', 'urgency'],
            'Someone_put_abag_on_gods_head': ['god', 'bag', 'head', 'divine', 'ridiculous'],
            'Dont_dream_it_be_it': ['dream', 'reality', 'transformation', 'aspiration', 'action'],
            'No_one_here_gets_out_alive': ['alive', 'mortality', 'escape', 'inevitable', 'death'],
            'These_things_were_secret_and_they_were_ours': ['secret', 'private', 'memory', 'possession', 'hidden'],
            'Every_man_is_aforged_man': ['forged', 'metal', 'creation', 'transformation', 'strength'],
            'This_machine_kills_fascists': ['machine', 'kill', 'political', 'resistance', 'weapon'],
            'Even_now_our_bodies_are_borderless': ['body', 'border', 'boundary', 'skin', 'limitless'],
            'In_heaven_everything_is_fine': ['heaven', 'divine', 'perfect', 'spiritual', 'paradise'],
            'This_page_intentionally_left_blank': ['page', 'blank', 'empty', 'intentional', 'document'],
            'You_wouldnt_download_afriend': ['download', 'digital', 'friend', 'ethics', 'technology'],
            'Abort_retry_fail': ['error', 'computer', 'system', 'failure', 'retry'],
            'We_are_the_dead_pixels': ['pixel', 'digital', 'dead', 'screen', 'corruption'],
            'There_is_no_cloud_just_other_peoples_computers': ['cloud', 'computer', 'illusion', 'technology', 'deception'],
            'Garbage_in_garbage_out': ['garbage', 'input', 'output', 'quality', 'computer'],
            'Hello_world_i_made_this': ['hello', 'world', 'creation', 'program', 'first'],
            'Your_fault_for_trusting_me': ['trust', 'betrayal', 'fault', 'responsibility', 'deception'],
            'The_internet_is_made_of_cats': ['internet', 'cat', 'animal', 'culture', 'meme'],
            'System_has_been_compromised': ['system', 'compromise', 'security', 'breach', 'danger'],
            'This_website_uses_cookies': ['website', 'cookie', 'internet', 'consent', 'tracking'],
            'If_you_see_this_it_worked': ['signal', 'success', 'confirmation', 'test', 'working'],
            'Clippy_watches_silently': ['clippy', 'surveillance', 'watching', 'silent', 'office'],
            'You_cant_fix_real_life_with_undo': ['undo', 'real', 'life', 'regret', 'irreversible'],
            'Please_do_not_power_off_during_update': ['power', 'update', 'system', 'warning', 'process'],
            'Welcome_user_everything_is_fine': ['welcome', 'user', 'fine', 'facade', 'deception'],
            '404hope_not_found': ['404', 'error', 'hope', 'lost', 'not_found'],
            'I_am_root_therefore_I_am': ['root', 'control', 'power', 'system', 'authority'],
            'The_singularity_will_be_memed': ['singularity', 'meme', 'future', 'technology', 'prophecy'],
            'I_am_the_state': ['state', 'power', 'authority', 'sovereignty', 'government'],
            'We_shall_fight_on_the_beaches': ['fight', 'beach', 'war', 'resistance', 'battle'],
            'Ask_not_what_your_country_can_do': ['country', 'service', 'patriotism', 'duty', 'nation'],
            'History_will_absolve_me': ['history', 'legacy', 'justice', 'time', 'judgment'],
            'The_crown_is_heavy_still': ['crown', 'heavy', 'burden', 'royalty', 'responsibility'],
            'Tear_down_this_wall': ['wall', 'tear', 'division', 'unity', 'destruction'],
            'Ihave_seen_the_promised_land': ['promised', 'land', 'vision', 'hope', 'future'],
            'My_body_politic': ['body', 'political', 'embodiment', 'representation', 'identity'],
            'Power_is_not_ameans_it_is_an_end': ['power', 'end', 'goal', 'domination', 'authority'],
            'The_only_thing_we_have_to_fear': ['fear', 'anxiety', 'terror', 'emotion', 'psychology'],
            'Give_me_liberty_or_give_me_death': ['liberty', 'freedom', 'death', 'choice', 'patriotism'],
            'By_the_sword_we_seek_peace': ['sword', 'peace', 'irony', 'weapon', 'conflict'],
            'Workers_of_the_world_unite': ['workers', 'unity', 'solidarity', 'labor', 'revolution'],
            'I_am_prepared_to_die': ['die', 'prepared', 'martyrdom', 'sacrifice', 'commitment'],
            'You_have_sat_too_long': ['sat', 'time', 'revolt', 'action', 'inactivity'],
            'I_am_the_first_accused': ['accused', 'justice', 'trial', 'first', 'responsibility'],
            'Please_hold_your_call_is_important': ['hold', 'call', 'waiting', 'important', 'patience'],
            'Dont_forget_the_milk_again': ['milk', 'forget', 'failure', 'reminder', 'routine'],
            'Laundry_is_my_only_religion': ['laundry', 'religion', 'routine', 'worship', 'daily'],
            'Woke_up_scrolled_back_to_sleep': ['wake', 'scroll', 'sleep', 'apathy', 'technology'],
            'Traffic_was_alesson_in_stillness': ['traffic', 'stillness', 'patience', 'lesson', 'meditation'],
            'Staring_into_the_fridge_like_god': ['fridge', 'stare', 'god', 'divine', 'everyday'],
            'Forgot_why_i_came_in_here': ['forgot', 'purpose', 'confusion', 'memory', 'human'],
            'Update_now_or_remind_me_later': ['update', 'remind', 'procrastination', 'choice', 'technology'],
            'Just_one_more_episode': ['episode', 'binge', 'denial', 'addiction', 'entertainment'],
            'This_meeting_couldve_been_an_email': ['meeting', 'email', 'frustration', 'efficiency', 'work'],
            'Your_package_is_on_the_way': ['package', 'delivery', 'expectation', 'waiting', 'commerce'],
            'So_many_tabs': ['tabs', 'overload', 'browser', 'information', 'chaos'],
            'Almost_did_yoga': ['yoga', 'almost', 'avoidance', 'intention', 'failure'],
            'Time_to_buy_toilet_paper_again': ['toilet', 'paper', 'repetition', 'necessity', 'routine'],
            'Eternal_recurrence_of_empty_subways': ['eternal', 'recurrence', 'subway', 'empty', 'transit'],
            'Do_androids_dream_of_dreaming': ['android', 'dream', 'uncertainty', 'artificial', 'consciousness'],
            'Clock_stuck_at_golden_hour': ['clock', 'golden', 'hour', 'suspension', 'aesthetic'],
            'Your_body_is_ahaunted_clock': ['body', 'haunted', 'clock', 'mortality', 'time'],
            'Today_we_choose_violent_meaninglessness': ['violent', 'meaningless', 'choice', 'absurdity', 'today'],
            'Yesterday_was_a_ghost_town': ['yesterday', 'ghost', 'town', 'loss', 'abandoned'],
            'Felt_cute_might_delete_later': ['cute', 'delete', 'confidence', 'regret', 'social_media'],
            'Being_is_alagging_buffer': ['being', 'lagging', 'buffer', 'latency', 'existence'],
            'Space_is_full_of_leftovers': ['space', 'leftovers', 'entropy', 'cosmic', 'waste'],
            'This_timeline_smells_like_regret': ['timeline', 'smell', 'regret', 'memory', 'past'],
            'Simulated_by_who_exactly': ['simulated', 'simulation', 'reality', 'question', 'existence'],
            'I_have_no_time_and_imust_scream': ['time', 'scream', 'panic', 'depression', 'urgency'],
            'We_left_earth_for_nothing': ['earth', 'left', 'nothing', 'emptiness', 'journey'],
            'Existence_is_weirdly_bugged_today': ['existence', 'bugged', 'glitch', 'error', 'reality'],
            'Hope_is_the_thing_with_feathers': ['hope', 'feathers', 'bird', 'lightness', 'poetry'],
            'I_am_large_icreep_my_multitudes': ['large', 'creep', 'multitudes', 'multiplicity', 'identity'],
            'The_leaves_were_full_of_children': ['leaves', 'children', 'nature', 'youth', 'poetry'],
            'I_too_sing_america_sometimes': ['sing', 'america', 'belonging', 'identity', 'nation'],
            'The_brain_is_wider_than_sky': ['brain', 'sky', 'mind', 'vastness', 'intelligence']
        }
        
        # Define bottom line keywords for enhanced matching
        self.bottom_line_keywords = {
            'Warning': ['warning', 'danger', 'caution', 'alert', 'threat', 'risk', 'peril', 'hazard', 'ominous', 'foreboding'],
            'Complicity': ['complicity', 'involvement', 'participation', 'collaboration', 'consent', 'agreement', 'silence', 'passive', 'accomplice'],
            'Compliance, surrender': ['compliance', 'surrender', 'submission', 'obedience', 'acquiescence', 'consent', 'agreement', 'silence', 'passive', 'yielding', 'capitulation'],
            'Ephemeral': ['ephemeral', 'temporary', 'fleeting', 'transient', 'momentary', 'brief', 'passing', 'short-lived', 'evanescent'],
            'Infinity': ['infinity', 'infinite', 'endless', 'eternal', 'limitless', 'boundless', 'unlimited', 'perpetual', 'timeless'],
            'Impermanance': ['impermanent', 'temporary', 'fleeting', 'transient', 'momentary', 'brief', 'passing', 'short-lived'],
            'Venom': ['venom', 'poison', 'toxic', 'deadly', 'lethal', 'harmful', 'dangerous', 'malicious', 'vicious'],
            'Absurdity': ['absurd', 'ridiculous', 'nonsensical', 'meaningless', 'pointless', 'futile', 'ludicrous', 'preposterous'],
            'Chaos': ['chaos', 'disorder', 'confusion', 'random', 'unpredictable', 'uncontrolled', 'anarchic', 'turbulent'],
            'Consequence': ['consequence', 'result', 'outcome', 'effect', 'impact', 'repercussion', 'aftermath', 'fallout'],
            'Awareness': ['awareness', 'consciousness', 'mindfulness', 'attention', 'perception', 'realization', 'understanding'],
            'Recurrence': ['recurrence', 'repetition', 'cycle', 'loop', 'return', 'repeat', 'recurring', 'cyclical'],
            'Seduction': ['seduction', 'temptation', 'allure', 'attraction', 'charm', 'enticement', 'lure', 'magnetism'],
            'Foreshadowing': ['foreshadow', 'omen', 'portent', 'sign', 'indication', 'hint', 'premonition', 'forewarning'],
            'Dread': ['dread', 'fear', 'terror', 'horror', 'anxiety', 'apprehension', 'foreboding', 'trepidation'],
            'Wonder': ['wonder', 'amazement', 'awe', 'marvel', 'astonishment', 'curiosity', 'fascination', 'mystery'],
            'Creation': ['creation', 'making', 'building', 'forming', 'constructing', 'generating', 'producing', 'fabricating'],
            'Desire': ['desire', 'longing', 'craving', 'yearning', 'passion', 'lust', 'want', 'need', 'appetite'],
            'Addiction': ['addiction', 'obsession', 'compulsion', 'dependence', 'habit', 'fixation', 'attachment'],
            'Transcendence': ['transcendence', 'elevation', 'ascension', 'spiritual', 'divine', 'sacred', 'holy', 'ethereal'],
            'Alienation': ['alienation', 'isolation', 'separation', 'loneliness', 'disconnection', 'estrangement', 'solitude'],
            'Imagery': ['imagery', 'visual', 'picture', 'image', 'scene', 'vision', 'appearance', 'representation'],
            'Insanity': ['insanity', 'madness', 'lunacy', 'derangement', 'mental', 'psychotic', 'delusional', 'crazy'],
            'Desperation': ['desperation', 'despair', 'hopelessness', 'urgency', 'need', 'crisis', 'emergency', 'distress'],
            'Divine': ['divine', 'godly', 'sacred', 'holy', 'spiritual', 'celestial', 'heavenly', 'religious'],
            'Liberation': ['liberation', 'freedom', 'release', 'emancipation', 'independence', 'autonomy', 'liberty'],
            'Mortality': ['mortality', 'death', 'dying', 'mortal', 'finite', 'temporary', 'human', 'fragile'],
            'Memory': ['memory', 'recollection', 'remembrance', 'nostalgia', 'past', 'history', 'recall', 'retrospect'],
            'Forging': ['forging', 'shaping', 'molding', 'forming', 'creating', 'building', 'constructing', 'fabricating'],
            'Resistance': ['resistance', 'opposition', 'defiance', 'rebellion', 'revolt', 'protest', 'struggle', 'fight'],
            'Skin': ['skin', 'surface', 'boundary', 'border', 'limit', 'edge', 'exterior', 'covering'],
            'Heaven': ['heaven', 'paradise', 'bliss', 'joy', 'happiness', 'divine', 'celestial', 'ethereal'],
            'Ethics': ['ethics', 'morality', 'values', 'principles', 'right', 'wrong', 'good', 'evil', 'virtue'],
            'Panic': ['panic', 'fear', 'terror', 'alarm', 'distress', 'anxiety', 'dread', 'horror'],
            'Corruption': ['corruption', 'decay', 'rot', 'degradation', 'deterioration', 'spoilage', 'contamination'],
            'Illusion': ['illusion', 'deception', 'false', 'fake', 'unreal', 'imaginary', 'phantom', 'mirage'],
            'Logic': ['logic', 'reason', 'rational', 'systematic', 'orderly', 'structured', 'analytical'],
            'Betrayal': ['betrayal', 'treachery', 'deceit', 'dishonesty', 'disloyalty', 'treason', 'duplicity'],
            'Culture': ['culture', 'society', 'community', 'tradition', 'custom', 'heritage', 'civilization'],
            'Breach': ['breach', 'break', 'violation', 'intrusion', 'penetration', 'infiltration', 'invasion'],
            'Consent': ['consent', 'permission', 'agreement', 'approval', 'acceptance', 'assent', 'compliance'],
            'Signal': ['signal', 'message', 'communication', 'indication', 'sign', 'mark', 'symbol'],
            'Surveillance': ['surveillance', 'monitoring', 'watching', 'observation', 'spying', 'tracking', 'oversight'],
            'Regret': ['regret', 'remorse', 'sorrow', 'guilt', 'shame', 'repentance', 'penitence'],
            'Suspense': ['suspense', 'tension', 'anticipation', 'uncertainty', 'anxiety', 'expectation', 'waiting'],
            'Facade': ['facade', 'appearance', 'surface', 'mask', 'disguise', 'pretense', 'illusion'],
            'Loss': ['loss', 'missing', 'gone', 'absent', 'deprived', 'bereft', 'without'],
            'Control': ['control', 'power', 'authority', 'dominance', 'command', 'influence', 'mastery'],
            'Prophecy': ['prophecy', 'prediction', 'forecast', 'omen', 'portent', 'vision', 'foresight'],
            'Sovereignty': ['sovereignty', 'authority', 'power', 'dominion', 'rule', 'control', 'supremacy'],
            'Service': ['service', 'duty', 'obligation', 'responsibility', 'commitment', 'devotion', 'loyalty'],
            'Legacy': ['legacy', 'heritage', 'inheritance', 'tradition', 'history', 'memory', 'remembrance'],
            'Burden': ['burden', 'weight', 'load', 'responsibility', 'obligation', 'duty', 'pressure'],
            'Division': ['division', 'separation', 'split', 'partition', 'segregation', 'isolation', 'disconnection'],
            'Vision': ['vision', 'sight', 'foresight', 'imagination', 'dream', 'aspiration', 'goal'],
            'Embodiment': ['embodiment', 'incarnation', 'manifestation', 'representation', 'personification', 'expression'],
            'Domination': ['domination', 'control', 'power', 'authority', 'supremacy', 'mastery', 'rule'],
            'Freedom': ['freedom', 'liberty', 'independence', 'autonomy', 'liberation', 'emancipation'],
            'Irony': ['irony', 'paradox', 'contradiction', 'opposite', 'unexpected', 'surprising', 'twist'],
            'Solidarity': ['solidarity', 'unity', 'togetherness', 'support', 'cooperation', 'alliance', 'partnership'],
            'Martyrdom': ['martyrdom', 'sacrifice', 'suffering', 'endurance', 'persecution', 'victimhood'],
            'Revolt': ['revolt', 'rebellion', 'revolution', 'uprising', 'insurrection', 'mutiny', 'protest'],
            'Justice': ['justice', 'fairness', 'equity', 'righteousness', 'law', 'order', 'truth'],
            'Waiting': ['waiting', 'patience', 'delay', 'pause', 'suspension', 'expectation', 'anticipation'],
            'Failure': ['failure', 'defeat', 'loss', 'mistake', 'error', 'fault', 'shortcoming'],
            'Routine': ['routine', 'habit', 'pattern', 'regular', 'daily', 'ordinary', 'normal'],
            'Apathy': ['apathy', 'indifference', 'unconcern', 'disinterest', 'passivity', 'lethargy'],
            'Patience': ['patience', 'tolerance', 'endurance', 'perseverance', 'forbearance', 'calmness'],
            'Divinity': ['divinity', 'godliness', 'sacredness', 'holiness', 'spirituality', 'celestial'],
            'Lack of purpose': ['purposeless', 'aimless', 'directionless', 'meaningless', 'pointless', 'futile'],
            'Procrastination': ['procrastination', 'delay', 'postponement', 'avoidance', 'hesitation', 'reluctance'],
            'Denial': ['denial', 'refusal', 'rejection', 'disbelief', 'negation', 'contradiction'],
            'Frustration': ['frustration', 'anger', 'irritation', 'annoyance', 'exasperation', 'disappointment'],
            'Expectation': ['expectation', 'anticipation', 'hope', 'prediction', 'forecast', 'assumption'],
            'Overload': ['overload', 'excess', 'overwhelm', 'surplus', 'overflow', 'abundance'],
            'Avoidance': ['avoidance', 'evasion', 'escape', 'shunning', 'ignoring', 'neglecting'],
            'Repetition': ['repetition', 'recurrence', 'reiteration', 'duplication', 'redundancy', 'monotony'],
            'Transit': ['transit', 'transportation', 'movement', 'journey', 'travel', 'passage'],
            'Uncertainty': ['uncertainty', 'doubt', 'confusion', 'ambiguity', 'indecision', 'hesitation'],
            'Suspension': ['suspension', 'pause', 'halt', 'stop', 'interruption', 'break'],
            'Absurdity': ['absurdity', 'nonsense', 'meaninglessness', 'pointlessness', 'futility', 'ridiculousness'],
            'Abandoned': ['abandoned', 'deserted', 'forsaken', 'neglected', 'discarded', 'left'],
            'Confidence': ['confidence', 'assurance', 'certainty', 'self-assurance', 'boldness', 'courage'],
            'Latency': ['latency', 'delay', 'lag', 'slowness', 'waiting', 'postponement'],
            'Entropy': ['entropy', 'disorder', 'chaos', 'decay', 'deterioration', 'disintegration'],
            'Remorse': ['remorse', 'regret', 'guilt', 'sorrow', 'penitence', 'repentance'],
            'Simulation': ['simulation', 'imitation', 'replica', 'copy', 'artificial', 'fake'],
            'Depression': ['depression', 'sadness', 'melancholy', 'despair', 'gloom', 'sorrow'],
            'Emptiness': ['emptiness', 'void', 'vacancy', 'hollowness', 'nothingness', 'blankness'],
            'Glitch': ['glitch', 'error', 'bug', 'malfunction', 'failure', 'corruption'],
            'Hope': ['hope', 'optimism', 'faith', 'belief', 'confidence', 'trust'],
            'Multiplicity': ['multiplicity', 'diversity', 'variety', 'plurality', 'manifold', 'various'],
            'Youth': ['youth', 'young', 'childhood', 'innocence', 'freshness', 'vitality'],
            'Belonging': ['belonging', 'inclusion', 'membership', 'affiliation', 'connection', 'attachment'],
            'Intelligence': ['intelligence', 'wisdom', 'knowledge', 'understanding', 'comprehension', 'insight']
        }

    def load_data(self):
        """Load the artistic analysis and hashtags data"""
        # Load artistic analysis for images2
        with open('../../image_analysis/images2_analysis/artistic_analysis_images2_filtered.json', 'r', encoding='utf-8') as f:
            self.artistic_analysis = json.load(f)
        
        # Load df.json for additional keywords
        with open('../../assets/DF.json', 'r', encoding='utf-8') as f:
            self.df_data = json.load(f)
        
        # Create a mapping from filename to df.json data
        self.df_mapping = {}
        for item in self.df_data:
            filename = item.get('filename', '')
            if filename:
                self.df_mapping[filename] = item
        
        # Load hashtags
        with open('Hashtags.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.hashtags = list(reader)
        
        print(f"Loaded {len(self.artistic_analysis)} images from images2, {len(self.df_data)} df.json entries, and {len(self.hashtags)} hashtags")

    def extract_quote_keywords(self, quote: str) -> List[str]:
        """Extract keywords from a quote automatically"""
        # Clean the quote
        clean_quote = quote.replace('_', ' ').lower()
        
        # Split into words and filter out common stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 
            'with', 'i', 'you', 'your', 'we', 'they', 'them', 'this', 'these', 'those',
            'have', 'had', 'do', 'does', 'did', 'am', 'been', 'being', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'shall', 'will', 'not', 'no', 'yes',
            'but', 'or', 'if', 'then', 'else', 'when', 'where', 'why', 'how', 'what',
            'who', 'whom', 'which', 'whose', 'my', 'me', 'myself', 'our', 'ours', 'ourselves'
        }
        
        # Extract meaningful words
        words = clean_quote.split()
        keywords = []
        
        for word in words:
            # Remove punctuation and clean the word
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 2 and clean_word not in stop_words:
                keywords.append(clean_word)
        
        return keywords

    def calculate_semantic_similarity(self, quote: str, image_data: dict, bottom_line: str = None) -> tuple[float, dict]:
        """Calculate semantic similarity between a quote and an image, returning score and detailed explanation"""
        score = 0.0
        explanation = {
            'direct_matches': [],
            'description_matches': [],
            'bottom_line_matches': [],
            'special_connections': [],
            'penalties': [],
            'bonuses': []
        }
        
        # Get quote-specific keywords (predefined + auto-extracted)
        predefined_keywords = self.quote_specific_keywords.get(quote, [])
        auto_keywords = self.extract_quote_keywords(quote)
        all_quote_keywords = list(set(predefined_keywords + auto_keywords))
        
        # Get image keywords and description
        image_keywords = [kw['keyword'].lower() for kw in image_data.get('keywords', [])]
        image_description = image_data.get('description', '').lower()
        
        # Add df.json keywords if available
        filename = image_data.get('filename', '')
        if filename in self.df_mapping:
            df_keywords = self.df_mapping[filename].get('keywords', [])
            df_keywords_lower = [kw.lower() for kw in df_keywords]
            image_keywords.extend(df_keywords_lower)
            
            # Also add artistic description from df.json
            df_description = self.df_mapping[filename].get('artistic_description', '')
            if df_description:
                image_description += ' ' + df_description.lower()
        
        # HIGHEST PRIORITY: Bottom line matching (if bottom_line is provided)
        if bottom_line and bottom_line in self.bottom_line_keywords:
            bottom_line_keywords = self.bottom_line_keywords[bottom_line]
            bottom_line_matches = []
            
            for keyword in bottom_line_keywords:
                if keyword in image_keywords:
                    score += 8.0  # Highest weight for bottom line keyword matches
                    bottom_line_matches.append(f"'{keyword}' (bottom line: {bottom_line})")
                elif keyword in image_description:
                    score += 6.0  # High weight for bottom line in description
                    bottom_line_matches.append(f"'{keyword}' in description (bottom line: {bottom_line})")
            
            if bottom_line_matches:
                explanation['bottom_line_matches'].extend(bottom_line_matches)
        
        # Check for direct keyword matches (high priority)
        for keyword in all_quote_keywords:
            if keyword in image_keywords:
                score += 5.0  # Very high weight for direct keyword matches
                explanation['direct_matches'].append(f"'{keyword}' found in image keywords")
            if keyword in image_description:
                score += 3.0  # Good weight for description matches
                explanation['description_matches'].append(f"'{keyword}' found in image description")
        
        # Special semantic connections based on keywords only (high priority for specific themes)
        special_connections = {
            'technology': ['computer', 'digital', 'screen', 'pixel', 'code', 'tech', 'machine', 'electronic', 'device', 'software', 'system', 'update', 'error', 'glitch', 'internet', 'cloud'],
            'nature': ['nature', 'earth', 'sky', 'tree', 'flower', 'plant', 'natural', 'organic', 'forest', 'garden', 'stars', 'moon', 'sun', 'animal'],
            'body': ['body', 'human', 'face', 'hand', 'person', 'head', 'skin', 'flesh', 'eye', 'mouth'],
            'death': ['skull', 'bone', 'grave', 'tomb', 'dead', 'mortal', 'corpse', 'cemetery', 'funeral'],
            'light': ['light', 'bright', 'glow', 'shine', 'illuminate', 'radiant', 'sun', 'lamp', 'bulb'],
            'dark': ['shadow', 'dark', 'black', 'night', 'obscure', 'hidden', 'dim', 'gloom'],
            'water': ['water', 'liquid', 'flow', 'ocean', 'river', 'drop', 'wave', 'sea', 'rain'],
            'fire': ['fire', 'flame', 'burn', 'heat', 'hot', 'blaze', 'smoke', 'ash'],
            'time': ['clock', 'time', 'hour', 'minute', 'temporal', 'moment', 'watch', 'timer'],
            'space': ['space', 'star', 'planet', 'cosmic', 'universe', 'galaxy', 'moon', 'sun'],
            'war': ['war', 'battle', 'fight', 'weapon', 'sword', 'violence', 'conflict', 'resistance'],
            'love': ['love', 'desire', 'romance', 'seduction', 'passion', 'heart', 'relationship'],
            'control': ['control', 'power', 'authority', 'state', 'government', 'crown', 'king', 'queen'],
            'memory': ['memory', 'secret', 'hidden', 'forgotten', 'past', 'history', 'archive'],
            'chaos': ['chaos', 'madness', 'insanity', 'confusion', 'disorder', 'random'],
            'fear': ['fear', 'dread', 'horror', 'terror', 'anxiety', 'panic', 'nightmare'],
            'hope': ['hope', 'faith', 'belief', 'heaven', 'divine', 'god', 'spiritual'],
            'reality': ['reality', 'illusion', 'dream', 'fantasy', 'simulation', 'virtual'],
            'routine': ['routine', 'daily', 'everyday', 'ordinary', 'normal', 'regular'],
            'absurdity': ['absurd', 'meaningless', 'nonsense', 'random', 'strange', 'weird'],
            'consent': ['signature', 'signatures', 'document', 'documents', 'agreement', 'contract', 'legal', 'official', 'formal', 'written', 'paper', 'manuscript', 'text', 'writing', 'letter', 'correspondence']
        }
        
        for theme, theme_words in special_connections.items():
            if any(word in quote.lower() for word in [theme] + theme_words):
                matching_words = [word for word in theme_words if word in image_keywords or word in image_description]
                if matching_words:
                    special_score = len(matching_words) * 2.0
                    score += special_score
                    explanation['special_connections'].append(f"Special '{theme}' connection: {', '.join(matching_words)} (+{special_score:.1f})")
        
        # Penalize very generic images that match everything
        if len(image_keywords) > 12:  # Very generic images with many keywords
            penalty = score * 0.4
            score -= penalty
            explanation['penalties'].append(f"Generic image penalty: too many keywords ({len(image_keywords)}) (-{penalty:.1f})")
        
        # Bonus for images with specific, relevant keywords
        specific_matches = sum(1 for kw in all_quote_keywords if kw in image_keywords)
        if specific_matches >= 2:
            bonus = specific_matches * 1.0
            score += bonus
            explanation['bonuses'].append(f"Multiple specific matches: {specific_matches} keywords (+{bonus:.1f})")
        
        return score, explanation

    def find_best_matches(self, quote: str, num_matches: int = 10, bottom_line: str = None) -> List[str]:
        """Find the best matching images for a given quote"""
        scores = []
        
        for image_data in self.artistic_analysis:
            score, _ = self.calculate_semantic_similarity(quote, image_data, bottom_line)
            if score > 0:  # Only include images with some relevance
                scores.append((image_data['filename'], score))
        
        # Sort by score (highest first) and take top matches
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Ensure we get exactly num_matches, but at least 8
        if len(scores) < 8:
            # If we don't have enough high-scoring matches, add some random ones
            all_images = [img['filename'] for img in self.artistic_analysis]
            used_images = set(score[0] for score in scores)
            remaining_images = [img for img in all_images if img not in used_images]
            random.shuffle(remaining_images)
            
            while len(scores) < 8 and remaining_images:
                scores.append((remaining_images.pop(), 0.1))  # Low score for random matches
        
        return [filename for filename, score in scores[:num_matches]]

    def find_best_matches_with_explanations(self, quote: str, num_matches: int = 10, used_images: set = None, bottom_line: str = None) -> List[tuple]:
        """Find the best matching images for a given quote with detailed explanations"""
        if used_images is None:
            used_images = set()
        
        scores_with_explanations = []
        
        for image_data in self.artistic_analysis:
            score, explanation = self.calculate_semantic_similarity(quote, image_data, bottom_line)
            # Only include images with meaningful semantic connections (score > 1.0)
            if score > 1.0:  # Higher threshold to exclude weak/random matches
                # Penalize images that have been used too much
                usage_penalty = 0
                if image_data['filename'] in used_images:
                    usage_penalty = 1.0  # Reduced penalty to allow more reuse
                
                adjusted_score = score - usage_penalty
                if adjusted_score > 0:  # Only include if still positive after penalty
                    scores_with_explanations.append((image_data['filename'], adjusted_score, explanation))
        
        # Sort by adjusted score (highest first)
        scores_with_explanations.sort(key=lambda x: x[1], reverse=True)
        
        # Take top matches, but ensure diversity
        selected_images = []
        for filename, adjusted_score, explanation in scores_with_explanations:
            if len(selected_images) >= num_matches:
                break
            selected_images.append((filename, adjusted_score, explanation))
            used_images.add(filename)
        
        # If we don't have enough high-scoring matches, use broader semantic matching
        if len(selected_images) < num_matches:
            remaining_needed = num_matches - len(selected_images)
            broader_matches = self.find_broader_semantic_matches(quote, remaining_needed, used_images, bottom_line)
            # Filter broader matches to only include those with meaningful scores
            meaningful_broader_matches = [(f, s, e) for f, s, e in broader_matches if s > 1.0]
            selected_images.extend(meaningful_broader_matches)
        
        # Remove random fallback - only show semantically connected images
        # If we still don't have enough matches, that's fine - better to show fewer relevant images
        
        return selected_images[:num_matches]
    
    def find_broader_semantic_matches(self, quote: str, num_needed: int, used_images: set, bottom_line: str = None) -> List[tuple]:
        """Find broader semantic matches when direct keywords fail"""
        quote_lower = quote.lower()
        broader_scores = []
        
        # Define broader semantic categories with more comprehensive keywords
        semantic_categories = {
            'emotion': ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'joy', 'sorrow', 'rage', 'anxiety', 'calm', 'peaceful', 'excited', 'worried', 'confused', 'satisfied', 'disappointed', 'proud', 'ashamed', 'grateful', 'melancholy', 'euphoria', 'despair', 'hope', 'love', 'hate', 'passion', 'apathy'],
            'texture': ['smooth', 'rough', 'soft', 'hard', 'shiny', 'dull', 'wet', 'dry', 'fuzzy', 'sharp', 'bumpy', 'flat', 'curved', 'straight', 'wavy', 'rippled', 'pitted', 'polished', 'matte', 'glossy', 'grainy', 'slippery', 'sticky', 'crunchy'],
            'size': ['big', 'small', 'large', 'tiny', 'huge', 'miniature', 'giant', 'microscopic', 'massive', 'petite', 'enormous', 'minuscule', 'colossal', 'diminutive', 'vast', 'compact', 'oversized', 'undersized'],
            'shape': ['round', 'square', 'triangle', 'circle', 'oval', 'rectangular', 'curved', 'straight', 'spherical', 'cylindrical', 'conical', 'pyramidal', 'hexagonal', 'octagonal', 'irregular', 'symmetrical', 'geometric', 'organic', 'angular'],
            'material': ['wood', 'metal', 'plastic', 'glass', 'fabric', 'stone', 'paper', 'leather', 'ceramic', 'concrete', 'brick', 'marble', 'granite', 'steel', 'aluminum', 'copper', 'gold', 'silver', 'iron', 'bronze', 'cotton', 'silk', 'wool'],
            'action': ['running', 'walking', 'jumping', 'flying', 'falling', 'rising', 'moving', 'standing', 'sitting', 'lying', 'climbing', 'swimming', 'dancing', 'singing', 'talking', 'listening', 'watching', 'reading', 'writing', 'drawing', 'painting', 'cooking', 'cleaning', 'working', 'playing', 'sleeping'],
            'time_of_day': ['morning', 'afternoon', 'evening', 'night', 'dawn', 'dusk', 'sunrise', 'sunset', 'midnight', 'noon', 'twilight', 'daybreak', 'nightfall', 'early', 'late', 'daytime', 'nighttime'],
            'weather': ['sunny', 'rainy', 'cloudy', 'stormy', 'windy', 'foggy', 'clear', 'overcast', 'snowy', 'hazy', 'misty', 'humid', 'dry', 'hot', 'cold', 'warm', 'cool', 'breezy', 'calm', 'turbulent'],
            'location': ['inside', 'outside', 'room', 'street', 'building', 'house', 'office', 'garden', 'park', 'forest', 'mountain', 'beach', 'city', 'country', 'suburb', 'downtown', 'neighborhood', 'community', 'kitchen', 'bedroom', 'bathroom', 'garage', 'basement', 'attic'],
            'concept': ['freedom', 'justice', 'peace', 'love', 'hate', 'truth', 'lies', 'beauty', 'ugliness', 'good', 'evil', 'right', 'wrong', 'success', 'failure', 'hope', 'despair', 'courage', 'fear', 'wisdom', 'ignorance', 'knowledge', 'learning', 'education', 'creativity', 'imagination'],
            'communication': ['speech', 'silence', 'voice', 'sound', 'noise', 'quiet', 'loud', 'whisper', 'shout', 'talk', 'listen', 'hear', 'speak', 'communicate', 'message', 'word', 'text', 'letter', 'language', 'conversation', 'dialogue', 'monologue'],
            'consent': ['agree', 'disagree', 'accept', 'reject', 'allow', 'deny', 'permit', 'forbid', 'consent', 'refuse', 'approve', 'disapprove', 'support', 'oppose', 'endorse', 'condemn', 'permission', 'authorization'],
            'science': ['science', 'laboratory', 'experiment', 'research', 'chemistry', 'biology', 'physics', 'microscope', 'test', 'analysis', 'discovery', 'innovation', 'invention', 'method', 'procedure', 'technique', 'scientific', 'hypothesis', 'theory', 'data', 'observation', 'measurement'],
            'technology': ['technology', 'computer', 'digital', 'electronic', 'device', 'machine', 'tool', 'equipment', 'gadget', 'appliance', 'system', 'software', 'hardware', 'network', 'internet', 'wireless', 'automated', 'mechanical', 'robotic'],
            'nature': ['nature', 'natural', 'organic', 'environmental', 'ecological', 'wild', 'untamed', 'pristine', 'wilderness', 'landscape', 'terrain', 'habitat', 'ecosystem', 'biodiversity', 'sustainability'],
            'art': ['art', 'artistic', 'creative', 'design', 'aesthetic', 'beautiful', 'elegant', 'stylish', 'fashionable', 'decorative', 'ornamental', 'expressive', 'imaginative', 'original', 'unique'],
            'music': ['music', 'musical', 'melody', 'rhythm', 'harmony', 'song', 'tune', 'sound', 'audio', 'acoustic', 'instrument', 'orchestra', 'band', 'concert', 'performance'],
            'food': ['food', 'eating', 'cooking', 'cuisine', 'meal', 'dish', 'ingredient', 'recipe', 'kitchen', 'restaurant', 'dining', 'nourishment', 'nutrition', 'delicious', 'tasty'],
            'transportation': ['transport', 'vehicle', 'car', 'bus', 'train', 'plane', 'bicycle', 'motorcycle', 'ship', 'boat', 'travel', 'journey', 'trip', 'commute', 'mobility'],
            'work': ['work', 'job', 'career', 'profession', 'occupation', 'employment', 'business', 'office', 'desk', 'computer', 'meeting', 'project', 'task', 'responsibility', 'duty']
        }
        
        for image_data in self.artistic_analysis:
            if image_data['filename'] in used_images:
                continue
                
            score = 0.0
            explanation = {'broader_semantic': []}
            
            image_keywords = [kw['keyword'].lower() for kw in image_data.get('keywords', [])]
            image_description = image_data.get('description', '').lower()
            
            # Check each semantic category
            for category, keywords in semantic_categories.items():
                category_matches = [kw for kw in keywords if kw in image_keywords or kw in image_description]
                if category_matches:
                    # Check if any category words appear in the quote
                    quote_matches = [word for word in keywords if word in quote_lower]
                    if quote_matches:
                        score += len(category_matches) * 2.0  # Much higher weight for semantic matches
                        explanation['broader_semantic'].append(f"Semantic category '{category}': {', '.join(category_matches)}")
                    
                    # Also check if the category name itself appears in the quote
                    if category in quote_lower:
                        score += len(category_matches) * 1.5
                        explanation['broader_semantic'].append(f"Quote mentions '{category}' category: {', '.join(category_matches)}")
            
            # Check for any word overlap between quote and image
            quote_words = set(self.extract_quote_keywords(quote))
            image_words = set(image_keywords)
            overlap = quote_words.intersection(image_words)
            
            if overlap:
                score += len(overlap) * 3.0  # Much higher weight for direct word matches
                explanation['broader_semantic'].append(f"Word overlap: {', '.join(overlap)}")
            
            # Check for bottom line matches in broader semantic matching
            if bottom_line and bottom_line in self.bottom_line_keywords:
                bottom_line_keywords = self.bottom_line_keywords[bottom_line]
                bottom_line_matches = [kw for kw in bottom_line_keywords if kw in image_keywords or kw in image_description]
                if bottom_line_matches:
                    score += len(bottom_line_matches) * 1.5  # High weight for bottom line in broader matching
                    explanation['broader_semantic'].append(f"Bottom line '{bottom_line}' matches: {', '.join(bottom_line_matches)}")
            
            # Check for partial word matches (e.g., "silence" matches "silent")
            for quote_word in quote_words:
                for image_word in image_words:
                    if len(quote_word) > 3 and len(image_word) > 3:
                        if quote_word in image_word or image_word in quote_word:
                            score += 1.0  # Higher weight for partial matches
                            explanation['broader_semantic'].append(f"Partial word match: '{quote_word}' ~ '{image_word}'")
            
            # Only include images with meaningful broader semantic connections
            if score > 1.0:
                broader_scores.append((image_data['filename'], score, explanation))
        
        # Sort by score and take top matches
        broader_scores.sort(key=lambda x: x[1], reverse=True)
        return broader_scores[:num_needed]

    def find_random_fallback_matches(self, quote: str, num_needed: int, used_images: set) -> List[tuple]:
        """This function is no longer used - we only show semantically connected images"""
        return []

    def create_connections(self):
        """Create semantic connections between quotes and images with detailed explanations"""
        print("Creating semantic connections with detailed explanations...")
        
        # Track used images to ensure diversity
        used_images = set()
        
        for hashtag in self.hashtags:
            quote = hashtag['Quote']
            quote_number = hashtag['Quote_number']
            bottom_line = hashtag['Bottom Line']
            origin = hashtag['Origin']
            
            # Find best matching images with explanations
            matching_images_with_explanations = self.find_best_matches_with_explanations(quote, num_matches=10, used_images=used_images, bottom_line=bottom_line)
            
            # Format the data for storage
            matching_data = []
            for filename, score, explanation in matching_images_with_explanations:
                # Find the strongest reason for this match (prioritize bottom line matches)
                strongest_reason = "No specific reason found"
                if explanation.get('bottom_line_matches'):
                    strongest_reason = explanation['bottom_line_matches'][0]
                elif explanation.get('direct_matches'):
                    strongest_reason = explanation['direct_matches'][0]
                elif explanation.get('special_connections'):
                    strongest_reason = explanation['special_connections'][0]
                elif explanation.get('description_matches'):
                    strongest_reason = explanation['description_matches'][0]
                elif explanation.get('broader_semantic'):
                    strongest_reason = explanation['broader_semantic'][0]
                elif explanation.get('random_fallback'):
                    strongest_reason = "Random selection (no strong semantic match)"
                
                matching_data.append({
                    'filename': filename,
                    'score': round(score, 2),
                    'reason': strongest_reason,
                    'full_explanation': explanation
                })
            
            # Store the connection
            self.quote_to_images[quote] = {
                'quote_number': quote_number,
                'quote': quote,
                'bottom_line': bottom_line,
                'origin': origin,
                'matching_images': [item['filename'] for item in matching_data],
                'matching_details': matching_data,
                'num_images': len(matching_data)
            }
        
        print(f"Created connections for {len(self.quote_to_images)} quotes")
        
        # Print diversity statistics
        all_used_images = set()
        for data in self.quote_to_images.values():
            all_used_images.update(data['matching_images'])
        
        total_possible_connections = len(self.quote_to_images) * 10
        actual_unique_connections = len(all_used_images)
        diversity_ratio = actual_unique_connections / total_possible_connections
        
        print(f"Diversity Statistics:")
        print(f"  Total possible image slots: {total_possible_connections}")
        print(f"  Unique images used: {actual_unique_connections}")
        print(f"  Diversity ratio: {diversity_ratio:.2%}")
        
        # Show most and least used images
        image_usage = {}
        for data in self.quote_to_images.values():
            for img in data['matching_images']:
                image_usage[img] = image_usage.get(img, 0) + 1
        
        most_used = sorted(image_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        least_used = sorted(image_usage.items(), key=lambda x: x[1])[:10]
        
        print(f"  Most used images: {most_used}")
        print(f"  Least used images: {least_used}")

    def save_connections(self, output_file: str = 'quote_to_images_connections.json'):
        """Save the connections to a JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.quote_to_images, f, indent=2, ensure_ascii=False)
        
        print(f"Saved connections to {output_file}")
        
        # Print some statistics
        total_images = sum(len(data['matching_images']) for data in self.quote_to_images.values())
        avg_images_per_quote = total_images / len(self.quote_to_images)
        print(f"Total images connected: {total_images}")
        print(f"Average images per quote: {avg_images_per_quote:.1f}")

    def print_sample_connections(self, num_samples: int = 5):
        """Print sample connections for verification"""
        print(f"\nSample connections (showing {num_samples}):")
        print("=" * 80)
        
        for i, (quote, data) in enumerate(list(self.quote_to_images.items())[:num_samples]):
            print(f"\n{i+1}. Quote: {quote}")
            print(f"   Bottom Line: {data['bottom_line']}")
            print(f"   Origin: {data['origin']}")
            print(f"   Images ({data['num_images']}): {', '.join(data['matching_images'][:5])}...")
            if len(data['matching_images']) > 5:
                print(f"   ... and {len(data['matching_images']) - 5} more")

def main():
    """Main function to run the semantic connector"""
    connector = SemanticHashtagConnector()
    
    # Load data
    connector.load_data()
    
    # Create connections
    connector.create_connections()
    
    # Save results
    connector.save_connections()
    
    # Print sample results
    connector.print_sample_connections()

if __name__ == "__main__":
    main() 