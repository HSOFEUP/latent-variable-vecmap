#!/bin/bash
#
# Copyright (C) 2019 Sebastian Ruder <sebastian@ruder.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA="$ROOT/data"
OUTPUT="$ROOT/output/"

METHOD_COUNT=1
METHOD_NAMES=('Latent-variable')
METHOD_TRAIN_ARGS=('--ruder_emnlp2018')

LANGUAGE_COUNT=3
LANGUAGE_SRCS=('en' 'en' 'en')
LANGUAGE_TRGS=('it' 'de' 'fi')
LANGUAGE_NAMES=('ENGLISH-ITALIAN' 'ENGLISH-GERMAN' 'ENGLISH-FINNISH')

DICTIONARY_COUNT=4
DICTIONARY_IDS=('5000' '25' 'numerals' 'identical')
DICTIONARY_NAMES=('5,000 WORD DICTIONARY' '25 WORD DICTIONARY' 'NUMERAL DICTIONARY' 'IDENTICAL STRINGS DICTIONARY')
DICTIONARY_SIZES=('5000' '25' '0' '0')
DICTIONARY_TRAIN_ARGS=('' '' '--init_numerals' '--init_identical')

SIMILARITY_DATASET_COUNT=2
SIMILARITY_DATASET_IDS=('mws353' 'rg65')
SIMILARITY_DATASET_NAMES=('MWS353' 'RG65')

for ((i = 0; i < $LANGUAGE_COUNT; i++))
do
    src=${LANGUAGE_SRCS[$i]}
    trg=${LANGUAGE_TRGS[$i]}
    echo '--------------------------------------------------------------------------------'
    echo ${LANGUAGE_NAMES[$i]}
    echo '--------------------------------------------------------------------------------'
    for ((j = 0; j < $DICTIONARY_COUNT; j++))
    do
        echo ${DICTIONARY_NAMES[$j]}
        for ((k = 0; k < $METHOD_COUNT; k++))
        do
            embedding_dir="$DATA/embeddings"
            output_dir="$OUTPUT/$src-$trg/${DICTIONARY_IDS[$j]}/${METHOD_NAMES[$k]}"
            mkdir -p "$output_dir"
            # adding cuda arguments; add verbose
            args="${METHOD_TRAIN_ARGS[$k]} ${DICTIONARY_TRAIN_ARGS[$j]} --precision fp32 -v"
            head -${DICTIONARY_SIZES[$j]} "$DATA/dictionaries/$src-$trg.train.shuf.txt" | python3 "$ROOT/map_embeddings.py" "$embedding_dir/$src.emb.txt" "$embedding_dir/$trg.emb.txt" "$output_dir/$src.emb.txt" "$output_dir/$trg.emb.txt" $args
            # evaluate translation with both nearest neighbour and CSLS retrieval
            echo -n "  - ${METHOD_NAMES[$k]}  |  Translation NN"
            python3 "$ROOT/eval_translation.py" --retrieval nn -d "$DATA/dictionaries/$src-$trg.test.txt" "$output_dir/$src.emb.txt" "$output_dir/$trg.emb.txt" | grep -Eo ':[^:]+%' | tail -1 | tr -d '\n'
            echo -n "  - ${METHOD_NAMES[$k]}  |  Translation CSLS"
            python3 "$ROOT/eval_translation.py" --retrieval csls -d "$DATA/dictionaries/$src-$trg.test.txt" "$output_dir/$src.emb.txt" "$output_dir/$trg.emb.txt" | grep -Eo ':[^:]+%' | tail -1 | tr -d '\n'
            for ((l = 0; l < $SIMILARITY_DATASET_COUNT; l++))
            do
                dataset="$DATA/similarity/$src-$trg.${SIMILARITY_DATASET_IDS[$l]}.txt"
                if [ -f "$dataset" ]
                then
                    echo -n "  ${SIMILARITY_DATASET_NAMES[$l]}"
                    python3 "$ROOT/eval_similarity.py" -l --backoff 0 "$output_dir/$src.emb.txt" "$output_dir/$trg.emb.txt" -i "$dataset" | grep -Eo ':[^:]+%' | tail -1 | tr -d '\n'
                fi
            done
            echo
        done
    done
    echo
done
