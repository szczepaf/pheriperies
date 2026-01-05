Discussion:

cities without trains:
Jesenice, Orlova

Duplicate cities: {'Jesenice': 3, 'Hranice': 3, 'Písek': 3, 'Chodov': 3, 'Sušice': 3, 'Hodonín': 3, 'Kyjov': 3, 'Benešov': 2, 'Kladno': 2, 'Milovice': 2, 'Říčany': 2, 'Tachov': 2, 'Žatec': 2}

But no has two times over 10k

Special notes:
Krupka - jen Krupka Bohosudov
Hranice - Hranice na Moravě

Budouci



vzdalenost od center jinych nez jen Kraje - tri urovne: Krajska, nad 200k (Praha, Brno, Ostrava) - jen Praha
multilinear regression: populace, volebni ucast, vzdalenost od Prahy (?)

Výběr měst:
- i mesta nad 5000 obyvatel
- vsechna mesta (náhodný vzorek)?

Dale:
- jine target variably nez jen volebni strany: zakomponovat prumerny prijem, nezamestnanost, etc.?
- decision trees - je distance dulezita promenna?

reporting: quarto


Úvod
Pojmem Periferie se označuje oblast vzdálená centru. Jedna z definic periferie v sociologickém slovníku je "přechod mezi městem a venkovem". https://encyklopedie.soc.cas.cz/w/Periferie. Ve veřejném diskurzu se o periferii často mluví v kontextu různých znevýhodnění a deprivací, poukazuje se na ztížené životní podmínky obyvatel periferie, špatný přístup ke zdravotnickým službám, horší ekonomickou situaci a s tím spojený odchod mladých lidí do center. V Česku se v souvislosti s tímto pojmem často hovoří o příhraniční oblasti zvané Sudety, kromě toho je ale zavedeným pojmem i termín "vnitřní periferie", který poukazuje na oblasti uvnitř krajů, které jsou vzdálené velkým městům a vykazují podobné znaky jako periferie příhraniční. https://www.irozhlas.cz/zpravy-domov/stredocesky-kraj-je-prestizni-volebni-trofej-uvnitr-je-ale-zaroven-mimoradne_2409020500_jgr

V souvislosti s letošními volbami do Poslanecké sněmovně se

https://www.irozhlas.cz/volby/volice-mobilizovala-hlavne-opozice-v-chudsim-prihranici-basty-vladnich-stran_2510051042_pik
řada analýz zabývala otázkou, zda periferie vykazují jisté společné znaky například co se týče složení odevzdaných hlasů v dané lokalitě. Ukazuje se např. na fakt, https://www.irozhlas.cz/volby/kdo-rozhodl-letosni-volby-obce-s-vyssim-podilem-nezamestnanych-a-lidi-se_2510051630_pik, že hnutí ANO v řadě oblastech vnitřní periferie výrazně posílilo, zatímco (možná poněkud překvapivě) hnutí stačilo naopak v těchto oblastech oslabilo.

Motivován těmito okolnostmi jsem se rozhodl na otázku periferií podívat optikou jednoho konkrétního parametru, a to sice dostupnosti vlakovou dopravou. Cílem této práce je zodpovědět otázku, zda odlehlost od center, uvažujeme-li spojení vlakovou sítí, je korelována s vyšším podílem odevzdaných hlasů v Poslanecké Sněmovní pro konrkétní politické subjekty.

Josef Bernard Nic se tady neděje


Dalsi testy:

a) Multiple linear regression (OLS)
Add more predictors, e.g.:

distance to regional city

population size / log(population)

turnout (UCAST_PROC)

region fixed effects (dummy vars for each regional city)

This lets you see whether “distance” still matters once you control for urbanisation, region, etc.

b) Polynomial / non-linear terms
If the effect of distance is not linear:

add 
distance
2
distance
2

or use splines (e.g. patsy / statsmodels with spline terms)

This tests, for example, whether the relationship is flat near the city and only rises in peripheral areas.


AVG age?