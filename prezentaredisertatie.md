# Prezentare Disertație

## 1. Context și obiectiv
Acest proiect propune o soluție completă pentru mentenanța predictivă bazată pe date temporale. Obiectivul principal este predicția duratei de viață rămasă (Remaining Useful Life - RUL) pentru motoare turbofan și identificarea anomaliilor în comportamentul acestor sisteme folosind datele NASA CMAPSS FD001.

Motivația vine din necesitatea industriei de a preveni opririle neplanificate și din creșterea eficienței operaționale prin intervenții planificate la timp. În disertație, punctez relevanța mentenanței predictive în aviație, fabricație și energie.

## 2. Alegerea datasetului și importanța sa
- Datasetul CMAPSS FD001 este un set standard acceptat în literatura academică pentru predictia RUL.
- Conține secvențe temporale simulate pentru motoare turbofan supuse uzurii și defectărilor.
- Folosește semnale provenite din senzori și variabile de condiție
totale, ceea ce permite testarea modelelor atât pe comportament normal, cât și pe situații anormale.
- Alegerea acestui dataset asigură comparabilitate cu alte studii și robustețe academică.

## 3. Arhitectura generală a proiectului
Structura proiectului este modulară și ușor de urmărit:
- `data/raw/` - datele originale descărcate de la NASA.
- `data/processed/` - datele rezultate din preprocesare.
- `src/` - codul sursă principal, împărțit pe module funcționale.
- `scripts/` - scripturi pentru generarea figurilor și calcule suplimentare.
- `app/` - aplicația Streamlit pentru vizualizarea rezultatelor.
- `outputs/` - artefacte rezultate (modele, predicții, rapoarte HTML).

Această organizare permite separarea clară a etapelor: încărcare, preprocesare, antrenare, inferență și evaluare.

## 4. Pașii esențiali ai pipeline-ului
### 4.1 Încărcare și pregătire date
- `src/data/cmapss_loader.py` citește fișierele text originale.
- Datele sunt transformate într-un DataFrame cu coloane pentru identificatorul unității, ciclul de viață și valorile senzorilor.
- Se păstrează coloanele relevante și se redenumesc dacă este necesar pentru lizibilitate.

### 4.2 Preprocesare și normalizare
- `src/data/preprocess.py` aplică standardizare pe caracteristici.
- Se elimină sau se impută datele lipsă, dacă sunt prezente.
- Se introduce o procedură de filtrare pentru a selecta subseturi normale atunci când se antrenează modelele de detecție a anomaliilor.
- Preprocesarea este necesară pentru a uniformiza scara caracteristicilor și a evita problemele de convergență ale rețelelor neuronale.

### 4.3 Generarea secvențelor temporale
- `src/models/lstm_rul.py` conține funcția `make_sequences()`.
- Datele sunt convertite în ferestre glisante de secvențe cu o lungime fixă (`seq_length`) și un pas configurabil (`step`).
- Pentru LSTM, modelul primește intrări de forma `[batch_size, seq_len, num_features]`.
- Predicția se face pe ultima fereastră, ceea ce înseamnă că modelul învață trenduri de degradare temporală.

### 4.4 Antrenarea și inferența modelelor
- `src/pipeline/run_all.py` coordonează întregul proces de la pregătire până la generarea rapoartelor.
- Pipeline-ul antrenează:
  - un model LSTM pentru regresia RUL;
  - un model XGBoost ca baseline pentru comparație;
  - un autoencoder pentru detecția anomaliilor.
- Rezultatele sunt salvate în `outputs/` și un raport HTML este generat pentru evaluare.

## 5. Modelul LSTM pentru predicția RUL
### 5.1 De ce LSTM?
- LSTM este o arhitectură de rețea neuronală recurentă care păstrează memoria pe termen lung.
- Este potrivită pentru date temporale cu dependențe între ciclurile succesive.
- Spre deosebire de perceptronul simplu, LSTM poate modela degradarea graduală a sistemului.

### 5.2 Implementarea actuală
- `src/models/lstm_rul.py` definește clasa `LSTMRegressor`.
- Arhitectura include:
  - `nn.LSTM` cu două straturi (`num_layers=2`);
  - dimensiunea ascunsă (`hidden_size=64`);
  - dropout între straturi pentru regularizare;
  - un head final `nn.Sequential` cu un strat liniar pentru output.
- Acest design oferă un compromis între complexitate și capacitatea de generalizare.

### 5.3 Detalii de antrenament
- Modelul este antrenat folosind o funcție de pierdere de regresie (de exemplu, MSE).
- Se folosesc batch-uri pentru stabilitate numerică și eficiență GPU/CPU.
- Implementarea include early stopping bazat pe pierderea de validare pentru a preveni suprainstruirea.

### 5.4 De ce nu un model mai simplu?
- Am ales LSTM în locul unui model bazat strict pe caracteristici statice pentru că datasetul conține informație temporală esențială.
- Alternativa simplă (de exemplu, regresie liniară sau XGBoost doar pe caracteristici instantanee) ar pierde semnalele de degradare progresivă.

## 6. Modelul de referință XGBoost
### 6.1 Rolul baseline-ului
- `src/models/baseline_xgb.py` antrenează un `xgboost.XGBRegressor`.
- XGBoost servește ca model de referință robust și performant.
- Comparația dintre LSTM și XGBoost este utilă pentru a demonstra avantajul abordării temporale față de un model de boosting pe caracteristici statice.

### 6.2 Avantaje XGBoost
- rapiditate și stabilitate la antrenare;
- bune capacități de generalizare pe seturi tabulare;
- interpretabilitate prin importanța caracteristicilor.

### 6.3 Limitări
- Nu captează în mod implicit dependențele secvențiale pe termen lung.
- Este sensibil la calitatea caracteristicilor extrase manual.

## 7. Modelul de detecție a anomaliilor
### 7.1 Alegerea autoencoderului
- `src/models/anomaly_autoencoder.py` implementează un autoencoder.
- Autoencoderul învață să reproducă intrările normate din comportamentul normal.
- Dacă o observație nu poate fi reconstructă precis, se consideră anormală.

### 7.2 Implementare și mecanism
- Modelul are un encoder și un decoder.
- Scopul este minimizarea erorii de reconstruire pentru datele normale.
- Se antrenează pe subseturi etichetate ca „normale”, astfel încât să nu învețe anomaliile.
- Pentru comparație, pipeline-ul generează și un model PCA, care este o metodă clasică de detecție a anomaliilor.

### 7.3 Compararea cu PCA
- PCA este o referință simplă și eficientă pentru detecția anomaliilor
- Autoencoderul oferă mai multă flexibilitate și capacitate de modelare a non-linearităților.
- Însă PCA este util pentru a verifica dacă problemele pot fi rezolvate și cu un model liniar.

### 7.4 Atenție la subsetul normal
- Implementarea curentă asigură că atât autoencoderul, cât și PCA sunt antrenate pe același subset normal.
- Aceasta este o corecție critică pentru a obține comparații echitabile între metode.

## 8. Evaluare și metrici
### 8.1 RUL
- Principalele metrici sunt eroarea medie și deviația de la RUL real.
- Se analizează distribuția erorilor pentru a identifica eventuale bias-uri și abateri sistematice.
- Rezultatele sunt salvate în `outputs/predictions/` și folosite în raportul HTML.

### 8.2 Anomalii
- Se calculează scoruri de reconstrucție pentru fiecare observație.
- Pragurile de alarmă sunt determinate empiric, comparând valorile normale și anormale.
- Se raportează rata de alarmă și sensibilitatea în `outputs/anomaly/anomaly_scores.csv`.

### 8.3 Raport final
- `outputs/report/report.html` conține grafice și statistici analizate automat.
- Acesta este materialul principal de analiză pentru susținerea disertației.

## 9. Structura codului și responsabilități
### 9.1 `src/data`
- `cmapss_loader.py` - încărcare și transformare raw.
- `features.py` - generare de trăsături și eventuale extra transformări.
- `preprocess.py` - normalizare, imputare și curățare.
- `split.py` - împărțirea pe seturi și transformare în secvențe.

### 9.2 `src/models`
- `lstm_rul.py` - modelul LSTM și funcțiile de generare a secvențelor.
- `baseline_xgb.py` - modelul XGBoost pentru baseline.
- `anomaly_autoencoder.py` - autoencoder și metoda PCA de detecție a anomaliilor.
- `explain_xgb.py` - funcții de interpretabilitate pentru XGBoost (dacă există).

### 9.3 `src/pipeline`
- `run_all.py` - orchestrare completă a procesului.
- `make_report.py` - generare de rapoarte și export.

### 9.4 `app/streamlit_app.py`
- Aplicație interactivă pentru explorarea rezultatelor.
- Permite încărcarea de predicții și afișarea de grafice.

## 10. Explicații tehnice detaliate
### 10.1 Preprocesare și normalizare
- Standardizarea caracteristicilor este necesară deoarece LSTM și autoencoderul funcționează mai bine când datele au media 0 și deviația standard 1.
- În practică, standardizatorul se potrivește pe datele de antrenament și se aplică apoi pe datele de validare/test.
- Acest lucru previne fugirea scării variabilelor și asigură stabilitate numerică.

### 10.2 Ferestre glisante și `step`
- Procesul de secvențiere împărțea datele pe fereastre de lungime fixă.
- Parametrul `step` determină cât de mult se deplasează fereastra pentru fiecare exemplu nou.
- Distanța mai mică între ferestre (pas mic) oferă mai multe exemple, dar crește suprapunerea și costul de calcul.
- Pasul `step=1` a fost folosit pentru a păstra informația temporală maximă.

### 10.3 Early stopping și validare
- Modelele neuronale pot suprainstrui rapid dacă antrenamentul rulează prea mult.
- Early stopping pe pierderea de validare oprește antrenamentul când performanța pe date necunoscute începe să scadă.
- Acest mecanism asigură generalizarea și este implementat în `src/models/lstm_rul.py` și `src/models/anomaly_autoencoder.py`.

### 10.4 Autoencoder vs PCA
- Autoencoderul este un model non-linea`r`
- PCA este un model liniar care reduce dimensiunea datelor pe componente principale.
- Autoencoderul poate captura structuri complexe de dependențe între senzori.
- PCA este util ca verficare simplă și pentru a arăta că datele conțin un subspațiu de comportament normal.

### 10.5 Crearea raportului
- Raportul HTML este generat automat din rezultatele obținute.
- Acesta include grafice de eroare, histograme și analize de performanță.
- Un astfel de raport este un element important pentru disertație, deoarece oferă dovezi vizuale și numerice ale calității soluției.

## 11. Trade-off-uri și justificări
### 11.1 De ce LSTM și nu Transformer?
- Transformerii sunt puternici pentru date secvențiale, dar necesită mai multe resurse și date mai mari.
- Pentru datasetul CMAPSS FD001, LSTM oferă un echilibru bun între performanță și complexitate.
- În viitor, un Transformer ar putea aduce îmbunătățiri, dar la costul unui timp de dezvoltare mai mare.

### 11.2 De ce XGBoost ca baseline?
- XGBoost este un model eficient pentru date tabulare și oferă o referință clară.
- Dacă LSTM nu ar aduce îmbunătățiri semnificative, modelul de boosted trees ar fi soluția practică preferată.
- Baseline-ul XGBoost justifică alegerea LSTM și arată că secvențialitatea oferă un avantaj real.

### 11.3 De ce autoencoder pentru anomalie?
- Autoencoderul permite modelarea comportamentului normal fără etichetare explicită a anomaliilor.
- În aplicații reale, anomaliile sunt rare și greu etichetate.
- Această alegere reflectă o situație industrială realistă.

### 11.4 Costuri vs. beneficii
- LSTM este mai costisitor decât un model simplu, dar aduce beneficii de acuratețe pentru date secvențiale.
- XGBoost este rapid și robust, dar limitat la caracteristici statice.
- Autoencoderul are cost de antrenare și necesita reguli pentru praguri de detecție, dar oferă o capacitate mai bună de identificare a deviațiilor.

## 12. Rezultate și ce pot prezenta
### 12.1 Rezultate obținute
- Predicțiile RUL pot fi comparate cu valorile reale și vizualizate în grafice.
- Scorurile de anomalii pot fi interpretate și folosite pentru generarea de alarme.
- Raportul final sumarizează această comparație și acoperă principalele concluzii.

### 12.2 Elemente cheie pentru prezentare
- Fluxul end-to-end: date raw → preprocesare → modele → evaluare.
- Relevanța industrială: reducerea downtime și prelungirea vieții utilajelor.
- Comparația model LSTM vs XGBoost: avantajul abordării temporale.
- Detecția anomaliilor: cum completează predicția RUL.

## 13. Ce am implementat concret
### 13.1 Funcții cheie
- `make_sequences()` în `src/models/lstm_rul.py` creează secvențe de intrare și etichete RUL.
- `train_autoencoder()` în `src/models/anomaly_autoencoder.py` antrenează modelul pe date normale și folosește validare pentru early stopping.
- `run_all.py` unifică pașii și scrie rezultatele în output.

### 13.2 Corecții aduse proiectului
- S-a asigurat că LSTM primește parametrul `step` corect la generarea secvențelor.
- S-a aliniat antrenamentul autoencoderului și PCA pe același subset normal.
- S-au adăugat tratări de excepție și instrucțiuni clare în aplicația Streamlit.

## 14. Extensii viitoare și recomandări
### 14.1 Îmbunătățiri de model
- testarea unui Transformer pentru RUL;
- evaluarea unei arhitecturi CNN-LSTM sau Temporal Convolutional Network;
- adăugarea de mecanisme de atenție pentru a identifica timpii importanți în secvență.

### 14.2 Extensii pentru anomalie
- folosirea unui model VAE (Variational Autoencoder);
- utilizarea unei rețele generative adversariale pentru anomalii;
- integrarea etichetelor de anomalii dacă sunt disponibile.

### 14.3 Structură de produs
i - implementarea unui API de inferență pentru integrare industrială;
- crearea unui modul de monitorizare continuă;
- exportul modelului într-un format ușor de servit (`ONNX`, `TorchScript`).

### 14.4 Analiză suplimentară
- analiza importanței caracteristicilor în XGBoost;
- explicabilitatea LSTM prin vizualizarea gradientelor sau attention maps;
- evaluarea pe alte seturi CMAPSS pentru generalizare.

## 15. Cum să folosești proiectul în prezentarea disertației
- Prezintă întâi problema și datasetul clasic CMAPSS.
- Explică succint fiecare etapă a pipeline-ului.
- Arată exemple de rezultate și grafice din `outputs/`.
- Sublinează avantajele încorporării atât a predicției RUL, cât și a detectării anomaliilor.
- Încheie cu lecțiile învățate și posibile extensii viitoare.

## 16. Concluzie
Acest proiect demonstrează o soluție robustă și reproductibilă pentru o problemă reală de mentenanță predictivă. Prin arhitectura LSTM + XGBoost + autoencoder, proiectul combină forța modelelor secvențiale cu simplitatea și eficiența baseline-ului tabular.

În disertație, subliniază modularitatea proiectului, motivarea fiecărei alegeri tehnice și importanța evaluării comparative. Astfel, proiectul nu este doar un set de modele, ci un pipeline complet gata pentru aplicații reale și pentru continuări viitoare.
