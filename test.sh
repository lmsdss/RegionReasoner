#!/bin/bash
# Run multi-turn evaluation on test sets.
# Usage: bash evaluation/eval_multi.sh <TEST_DATA_JSON> <MODEL_DIR>

# RefCOCOg multi-turn test
bash evaluation/eval_multi.sh /RegionReasoner/test/testdata/refcocog_multi_turn.json /RegionReasoner/test/pretrained_models/RegionReasoner-7B

# RefCOCO+ multi-turn test
bash evaluation/eval_multi.sh /RegionReasoner/test/testdata/refcocoplus_multi_turn.json /RegionReasoner/test/pretrained_models/RegionReasoner-7B

