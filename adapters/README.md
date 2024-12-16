### Config files and parameters of the adapters.


### Section 3.1
Bottleneck Adapters, CNet: 
* jekunz/bottleneck4
* jekunz/bottleneck16
* jekunz/bottleneck64

LoRA Feed-forward, CNet: 
* jekunz/lora256ff2
* jekunz/lora128ff
* jekunz/lora64ff
* jekunz/lora32ff
* jekunz/lora8ff

LoRA Attention, CNet: 
* jekunz/lora1024attn
* jekunz/lora256attn
* jekunz/lora128attn
* jekunz/lora32attn
* jekunz/lora8attn

(IA)^3: 
* jekunz/ia3

Prefix Tuning: 
* jekunz/prefix32

### Section 3.2
Ablations on LoRA modules: 
* jekunz/lora256ff2
* jekunz/lora256attn
* jekunz/lora256all

### Section 3.3
Ablations on layer exclusion: 
* jekunz/lora32-allbutlast2
* jekunz/lora32-allbutlast4
* jekunz/lora32-onlylast2
* jekunz/lora32-onlylast4

### Section 3.4
Ablations with Icelandic Gigaword Corpus (IGC): 
* jekunz/bottleneck4-igc
* jekunz/lora256ff-igc
