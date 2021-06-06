# tinyml-speech

* _Featue___Extraction.py_	 - skrypt służy do wyodrębnienia cech ze zbioru danych, w tym przypadku będą to zapisane MFCC, oprócz pozyskiwania cech z całego datasetu, tworzyłem też zbiory cech zbiorów czysto testowych - były to próbki słowa "kakapo" osób trzecich, dostępne w folderze **_TestVoices_**
* _TrainTestMmodel.py_ - Trenowanie modelu oraz jego ewaluacja, za pomocą zmiennej **czy_model** można ustalić, czy trenujemy nowy model, czy też testujemy model już istniejący (na nowym zestawie danych)