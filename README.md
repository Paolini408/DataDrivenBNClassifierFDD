# Analysis of an experimental HVAC system with fully data-driven Bayesian Networks

## Indice
1. [Introduzione al Caso Studio](#introduzione-al-caso-studio)
2. [Descrizione del Dataset](#descrizione-del-dataset)
3. [Selezione delle Variabili di Input](#selezione-delle-variabili-di-input)
4. [Modelli data-driven di Classificazione](#modelli-data-driven-di-classificazione)
5. [Conclusioni](#conclusioni)

---

## Introduzione al Caso Studio
Questo progetto esplora il funzionamento di tre diversi modelli di classificazione basati su reti bayesiane applicati a un sistema HVAC sperimentale.
Il sistema analizzato è un'UTA a volume d'aria costante a doppio ventilatore e singolo condotto.
Esso è progettato per controllare la temperatura dell'aria, l'umidità relativa, la velocità dell'aria e la qualità dell'aria all'interno di una stanza di test.

Il layout dell'impianto, le variabili monitorate e le logiche di controllo del sistema sono rappresentate nella seguente immagine:

![AHU Layout](figs/AHU_layout.png)

---

## Descrizione del Dataset
Il dataset, in formato .CSV, è già stato preprocessato e contiene dati etichettati per condizioni normali e diverse tipologie di guasti.
Le misurazioni sono state acquisite con un timestep di 1 minuto. Inoltre, le potenze termiche delle due batterie sono state calcolate utilizzando la portata e il delta T, mentre gli stati on/off dei componenti sono stati derivati in base alle logiche di controllo precedentemente definite.

---

## Selezione delle Variabili di Input
La selezione delle variabili di input è un passaggio fondamentale e il numero di queste ultime può essere impostato dall'utente.
Il criterio utilizzato per scegliere le variabili più rilevanti è basato sull'indice di mutua informazione (MI score).

- Nel caso di modelli continui, viene utilizzato un numero di bin estremamente elevato.
- Per modelli discreti, viene applicata una discretizzazione in base a un numero di bin scelto dall'utente.

---

## Modelli data-driven di Classificazione
Sono stati implementati tre diversi modelli di classificazione basati su reti bayesiane:

### 1) Conditional Gaussian Network (CGN)
Il modello assume che le variabili continue seguano una distribuzione gaussiana condizionata sulle etichetto di guasto (o normale).
- Maggiori dettagli: [CGN classifier with probabilistic boundary](https://www.sciencedirect.com/science/article/pii/S1359431116310675)

### 2) Kernel Density Estimation (KDE)
Questo modello utilizza una stima della densità a kernel per approssimare la distribuzione delle variabili continue, senza fare ipotesi parametriche.
- Maggiori dettagli: [FDD using Interpolated Kernel Density Estimate](https://www.sciencedirect.com/science/article/pii/S0263224121002438)

### 3) Cost-Sensitive Tree Augmented Naive Bayesian Network (TAN)
Un'estensione del modello Naive Bayes che introduce dipendenze tra i sintomi (variabili selezionate), migliorando la capacità di classificazione.
Inoltre, grazie alla possibilità di definire un peso per ogni classe, il modello è in grado di gestire meglio situazioni di sbilanciamento.

- Maggiori dettagli: [Discrete Bayesian Networks](https://www.sciencedirect.com/science/article/pii/S0140700719301070), [Cost-sensitive Bayesian network classifiers](https://www.sciencedirect.com/science/article/pii/S0167865514001354)

---

## Conclusioni
Questo progetto offre un'analisi delle prestazioni di diversi modelli di classificazione bayesiani data-driven applicati a un sistema HVAC.

Per ulteriori dettagli, contatta il creatore Marco Paolini.
