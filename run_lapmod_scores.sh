#!/bin/bash
#
# Copyright (C) 2017  Mikel Artetxe <artetxem@gmail.com>
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
OUTPUT="$ROOT/output/acl2017"

METHOD_COUNT=1
#METHOD_IDS=('mlp' 'mikolov2013a' 'xing2015' 'zhang2016' 'artetxe2016' 'artexte2017')
METHOD_IDS=('LAPMOD_similar_3_repeat_2')
#METHOD_NAMES=('MLP' 'Mikolov et al. (2013a)' 'Xing et al. (2015)    ' 'Zhang et al. (2016)   ' 'Artetxe et al. (2016) ' 'Artexte et al. (2017)       ')
METHOD_NAMES=('LAPMOD_similar_3_repeat_2')
#METHOD_TRAIN_ARGS=('--mlp --self_learning' '--unconstrained' '--orthogonal' '--orthogonal' '--orthogonal' '--orthogonal --self_learning')
METHOD_TRAIN_ARGS=('--orthogonal --self_learning --lapmod --n-similar 3 --lap-repeats 2')
#METHOD_EVAL_ARGS=('--dot' '' '' '' '' '--dot')
METHOD_EVAL_ARGS=('--dot')
#METHOD_EMBEDDINGS=('unit-center' 'original' 'unit' 'original' 'unit-center' 'unit-center')
METHOD_EMBEDDINGS=('unit-center')

LANGUAGE_COUNT=1
#LANGUAGE_SRCS=('en' 'en' 'en')
#LANGUAGE_SRCS=('en' 'en')
LANGUAGE_SRCS=('en')
#LANGUAGE_TRGS=('it' 'de' 'fi')
#LANGUAGE_TRGS=('de' 'fi')
LANGUAGE_TRGS=('it')
#LANGUAGE_NAMES=('ENGLISH-ITALIAN' 'ENGLISH-GERMAN' 'ENGLISH-FINNISH')
#LANGUAGE_NAMES=('ENGLISH-GERMAN' 'ENGLISH-FINNISH')
LANGUAGE_NAMES=('ENGLISH-ITALIAN')

#DICTIONARY_COUNT=3
DICTIONARY_COUNT=1
#DICTIONARY_COUNT=2
#DICTIONARY_IDS=('5000' '25' 'numerals')
DICTIONARY_IDS=('5000')
#DICTIONARY_IDS=('25' 'numerals')
#DICTIONARY_NAMES=('5,000 WORD DICTIONARY' '25 WORD DICTIONARY' 'NUMERAL DICTIONARY')
DICTIONARY_NAMES=('5,000 WORD DICTIONARY')
#DICTIONARY_NAMES=('25 WORD DICTIONARY' 'NUMERAL DICTIONARY')
#DICTIONARY_SIZES=('5000' '25' '0')
DICTIONARY_SIZES=('5000')
#DICTIONARY_SIZES=('25' '0')
#DICTIONARY_TRAIN_ARGS=('' '' '--numerals')
DICTIONARY_TRAIN_ARGS=('')
#DICTIONARY_TRAIN_ARGS=('' '--numerals')

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
            embedding_dir="$DATA/embeddings/${METHOD_EMBEDDINGS[$k]}"
            output_dir="$OUTPUT/$src-$trg/${DICTIONARY_IDS[$j]}/${METHOD_IDS[$k]}"
            mkdir -p "$output_dir"

            # adding cuda arguments; add verbose
            args="${METHOD_TRAIN_ARGS[$k]} ${DICTIONARY_TRAIN_ARGS[$j]} --precision fp32 -v"
            head -${DICTIONARY_SIZES[$j]} "$DATA/dictionaries/$src-$trg.train.shuf.txt" | python3 "$ROOT/map_embeddings.py" "$embedding_dir/$src.emb.txt" "$embedding_dir/$trg.emb.txt" "$output_dir/$src.emb.txt" "$output_dir/$trg.emb.txt" $args
            echo -n "  - ${METHOD_NAMES[$k]}  |  Translation"
            python3 "$ROOT/eval_translation.py" ${METHOD_EVAL_ARGS[$k]} -d "$DATA/dictionaries/$src-$trg.test.txt" "$output_dir/$src.emb.txt" "$output_dir/$trg.emb.txt" | grep -Eo ':[^:]+%' | tail -1 | tr -d '\n'
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
