
<hr>

**UPDATE**

The model (`arabic-news.tar.gz`) consists of pre-trained embedding vectors trained on the `arabic-news.txt` corpus.
Both the model and the dataset are available for download on HuggingFace [arabic-news-embeddings](https://huggingface.co/azizalto/arabic-news-embeddings).

<hr>

<!--
arabic-news.tar.gz (download [here](https://drive.google.com/open?id=0ByiDbCx0i9pEQV9ZUEFIb0hwMmM)) is a pre-trained vectors trained on the arabic-news.txt corpus (download [here](https://drive.google.com/open?id=0ByiDbCx0i9pESG5vRG05aXJ4TFU)).
-->

It contains around 159K vocabulary (of dimension 300).

File size:
	compressed 170.94MB (and 193.2MB un-compressed)

Trained on word2vec, using the following command:

`$ ./word2vec -train $CORPUS -output $OUT -cbow 1 -size 300 -window 10 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15`


NOTE: If you use this, pls cite the paper. Many thanks!
```
@inproceedings{altowayan2016word,
  title={Word embeddings for Arabic sentiment analysis},
  author={Altowayan, A. Aziz and Tao, Lixin},
  booktitle={Big Data (Big Data), 2016 IEEE International Conference on},
  pages={3820--3825},
  year={2016},
  organization={IEEE}
}
```


### Sample queries:

Analogy

```
$./word2vec/word-analogy arabic-news.bin
Enter three words (EXIT to break): السعودية الرياض اليابان

Word: السعودية  Position in vocabulary: 243
Word: الرياض  Position in vocabulary: 1765
Word: اليابان  Position in vocabulary: 1509

                                              Word              Distance
------------------------------------------------------------------------
                                        	طوكيو		0.533433
                                  			سنغافورة		0.452111
                                          	بكين		0.447027
                                        	الصين		0.416526
                                        	سيئول		0.380522
                              				تركمانستان		0.364201
                                  			الياباني		0.359618

```
```
Enter three words (EXIT to break): اميركا واشنطن الصين

Word: اميركا  Position in vocabulary: 4813
Word: واشنطن  Position in vocabulary: 577
Word: الصين  Position in vocabulary: 569

                                              Word              Distance
------------------------------------------------------------------------
                                          بكين		0.630212
                                        طهران		0.547874
                                    الصينية		0.529155
                                        إيران		0.520291
                                        للصين		0.490572
                                      تايوان		0.490066
                                        روسيا		0.471511
```
```
Enter three words (EXIT to break): ملك رجل ملكة

Word: ملك  Position in vocabulary: 2979

Word: رجل  Position in vocabulary: 468

Word: ملكة  Position in vocabulary: 11277

                                              Word              Distance
------------------------------------------------------------------------
                                      متزوجة		0.350069
                                          لرجل		0.336586
                                        امرأة		0.318018
                                            شاب		0.315098
                                          سيدة		0.308970
                                          فتاة		0.305358
                                    والرائح		0.297582
                                        إمرأة		0.293696

```


And more sample queries

On word distance:

```

$ ./word2vec/distance arabic-news.bin
Enter word or sentence (EXIT to break): جميل

Word: جميل  Position in vocabulary: 450

                                              Word       Cosine distance
------------------------------------------------------------------------
                                          رائع		0.758100
                                          ممتع		0.737023
                                          بسيط		0.637770
                                            شيق		0.623244
                                            سلس		0.620555
                                    راااائع		0.601265
                                          ماتع		0.599824
```

```
Enter word or sentence (EXIT to break): سخيف

Word: سخيف  Position in vocabulary: 25908

                                              Word       Cosine distance
------------------------------------------------------------------------
                                          تافه		0.495980
                                          سطحي		0.444503
                                        سخيفة		0.443913
                                        مبتذل		0.438295
                                            سيء		0.433009
                                            ممل		0.432163
											عادي		0.414368
```

```
Enter word or sentence (EXIT to break): جيد

Word: جيد  Position in vocabulary: 893

                                              Word       Cosine distance
------------------------------------------------------------------------
                                        ممتاز		0.623296
                                          مميز		0.512442
                                          جميل		0.493209
                                        مناسب		0.486915
                                          كبير		0.482421
                                          مشوق		0.479037
                                            خاص		0.466387

```



