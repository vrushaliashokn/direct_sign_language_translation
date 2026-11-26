# Direct Sign Language Translation

Sign language as a medium of communication is incredibly complex. Most signing systems are not simply direct translations of spoken word, but consist of their own intricate syntax and rules. The question of sign language translation poses many challenges: signs are very sensitive to qualities like the shape of the hand, orientation of the palm in a 3D-space, or the speed of the movement. Signs are also completed through facial expressions or other non-hand signals that add to the complexity of the task.

Contrastive learning is a powerful technique that can associate data of different modalities in a shared embedding space, commonly used to learn similarities between different augmentations of the same image or images and their associated textual captions. Such an approach would eliminate the need for translation into a gloss intermediary before subsequent generation in the target language. This project exploits Contrastive Learning’s affinity to associate cross-modal data in the field of Sign Language Translation between two distinct signing systems.

The main contributions of this project can be described as follows:
- An architecture for word-level Sign Language Recognition on Indian Sign Language (ISL) and American Sign Language (ASL), with accuracies reaching 90% and 80% respectively.
- Exploration and implementation of a Contrastive Learning based translation framework that can complete cross-system translation in both directions (i.e. ISL to ASL (63% accurate) and ASL to ISL (68% accurate).

# Datasets
[INCLUDE](https://huggingface.co/datasets/ai4bharat/INCLUDE) is a comprehensive word-level ISL dataset. Signs consist of the Tamil variations of ISL, with over 4292 videos for 263 word signs from 15 different word categories, signed by 7 different signers from St. Louis School for the Deaf in Chennai. The best performing model achieves an accuracy of 94.5% on the INCLUDE-50 dataset and 85.6% on the INCLUDE dataset. Videos in this dataset are very clean and of good clarity. Each sample corresponds to a single word.

[ASLLRP](https://dai.cs.rutgers.edu/dai/s/dai) is originally a sentence-level dataset for ASL provided by Boston University. The dataset also provides a word-level corpus, created from cropping word instances from the original source sentence videos. The dataset, with over 9,674 videos covers over 3,300 ASL signs, including compound signs, by 1-6 experienced signers.
