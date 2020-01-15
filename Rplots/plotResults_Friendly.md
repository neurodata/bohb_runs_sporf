---
title: "BOHB Run on OpenML-100-Freindly"
author: "Jesse Leigh Patsolic"
output: 
  html_document:
    toc: true
    self_contained: true
    keep_md: true
---

<!--
### ### INITIAL COMMENTS HERE ###
###
### Jesse Leigh Patsolic 
### 2020 <jpatsol1@jhu.edu>
### S.D.G 
#
-->



<style type="text/css">
.table {
    width: 40%;
}
tr:hover {background-color:#f5f5f5;}
</style>


# BOHB results as run on Synaptomes1:

Using the datasets from the [OpenML Study 225 -- Friendly
100](https://www.openml.org/s/225) we compare the results of sklearn's
RF and NeuroData's SPORF.  


Each dataset is partitioned into `training`, `validation`, and `testing`
sets.  Using [BOHB via HPBandSter](https://github.com/automl/HpBandSter)
the classifiers are trained on the `training` set and the loss is
calculated on the `validation` set for parameter tuning (BOHB) and the
loss on the `testing` set is reported in the plots and when determining if
SPORF does as well as sklearn's RF.

The hyper-parameter search space is given below:
```
Configuration space object:

Hyperparameters:
  clf, Type: Categorical, Choices: {skrf, sporf}, Default: skrf                                                 
  max_depth, Type: UniformInteger, Range: [2, 65535], Default: 362, on log-scale                                
  max_features_sk, Type: UniformFloat, Range: [0.01, 0.9], Default: 0.455                                       
  max_features_sporf, Type: UniformFloat, Range: [0.01, 4.0], Default: 2.005                                    
  sporf_fc, Type: Categorical, Choices: {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 8.0}, Default: 1.0                       
Conditions:
  max_features_sk | clf == 'skrf'
  max_features_sporf | clf == 'sporf'
  sporf_fc | clf == 'sporf'

```


```r
#dataDir <- "../output_syn1_100f/"
dataDir <- "../results_openml_s_225/output_syn1_202001/"

a0 <- dir(dataDir, full.names = TRUE)

f <- grep("csv", a0, value = TRUE)[-31]
## dataID 1515 caused problems.


ids <- as.integer(sapply(f, function(x) strsplit(tail(strsplit(x ,"_")[[1]], 1), ".csv")[[1]]))
names(ids) <- NULL


DAT <- lapply(f, fread)

DAT <- lapply(DAT, function(dat){
	mf1 <- dat$max_features_sporf
	mf2 <- dat$max_features_sk
	mf1[is.na(mf1)]<-mf2[is.na(mf1)]
	dat$mf <- mf1
	dat[, c("V1", "max_features_sk", "max_features_sporf") := NULL]
	dat}
)

for(i in 1:length(DAT)){
	DAT[[i]]$id <- ids[[i]]
}

suppressWarnings({
md <- lapply(DAT, melt)
})

margin <- sapply(DAT, function(dat){ -1*diff(aggregate(dat[, .(test_loss)], list(dat$clf), FUN = min)[, 2]) })

marDiff <- data.table(margin, ids)
```

Using [OpenML Study 225 -- Friendly 100](https://www.openml.org/s/225) a
friendly version of OpenML 100, the following data IDs were run:

6, 11, 12, 14, 16, 18, 22, 28, 32, 36, 37, 44, 54, 60, 182, 300, 375, 458, 554, 1038, 1049, 1050, 1063, 1067, 1068, 1120, 1459, 1466, 1467, 1468, 1471, 1475, 1476, 1478, 1479, 1485, 1487, 1489, 1491, 1492, 1493, 1494, 1497, 1501, 1504, 1510, 4134, 4538, 40499.


## A random sample of runs plotted with jittered violin plots


```r
p <- list()
j <- 1
samp <- sample(length(DAT), 9)

samp <- c(19, samp[-1])
for(i in samp){
#for(i in 1:length(DAT)){
	dat <- DAT[[i]]

	minLoss <- aggregate(dat, list(dat$clf), min)

	poss <- 
		suppressWarnings(
			aggregate(dat, list(dat$clf), median)
		)


	p[[j]] <- 
		ggplot(data = dat, 
					 aes(x = clf, y = test_loss, color = clf, group = clf)) + 
			  geom_violin(alpha = 0.3) + 
			  geom_jitter(size = max(dat$budget/max(dat$budget),0.1), height = 0) +
			  geom_text(data = minLoss, aes(label = paste0("min = ",round(test_loss,3)), y = poss[[6]]), alpha = 0.5, colour = "black") + 
			  ggtitle(paste0("openml_d:", ids[i]))
	j <- j + 1
}

p$ncol = 3
do.call(grid.arrange, p)
```

![](plotResults_Friendly_files/figure-html/unnamed-chunk-1-1.png)<!-- -->

<table>
 <thead>
  <tr>
   <th style="text-align:left;"> Links_to_datasets </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> [1479](https://openml.org/d/1479) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> [1067](https://openml.org/d/1067) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> [1489](https://openml.org/d/1489) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> [28](https://openml.org/d/28) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> [1493](https://openml.org/d/1493) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> [1068](https://openml.org/d/1068) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> [14](https://openml.org/d/14) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> [4538](https://openml.org/d/4538) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> [1501](https://openml.org/d/1501) </td>
  </tr>
</tbody>
</table>


## A histogram showing the differnce in loss, i.e. ((loss(SKRF) - loss(SPORF))


```r
win <- sum(marDiff$margin > 0)
loss <- sum(marDiff$margin < 0)
tie <- sum(marDiff$margin == 0)
tot <- nrow(marDiff)

hist(marDiff$margin,breaks = "fre", 
	 main = sprintf("positive: sporf won %d/%d, negative: sporf lost %d/%d tied: %d/%d\n intervals are closed on the left, open on right", win, tot, loss, tot, tie, tot), 
	 right = FALSE, xlab = "difference (RF - Sporf)")
abline(v = 0.0, col = 'red', lwd =2)
```

![Histogram showing the difference in loss between the two algorithms.](plotResults_Friendly_files/figure-html/unnamed-chunk-3-1.png)

```r
#
#hist(marDiff$margin,breaks = "fre", 
#	 main = "positive: sporf won,\n negative: sporf lost\n intervals are open on the left, closed on right", 
#	 right = TRUE, xlab = "difference (RF - Sporf)")
#abline(v = 0.0, col = 'red', lwd =2)
```

### A different view of the above plot


```r
tmp <- marDiff
tmp$SporfOutcome <- ifelse(tmp$margin < 0, "lost", "won")
tmp$SporfOutcome[tmp$margin == 0] <- "tied"


link_id <- sprintf("[%d](%s%d)", ids[which(tmp$margin > 0.05)], "https://openml.org/d/", ids[which(tmp$margin > 0.05)])

colorLegend <- c("darkorange", "black", "darkmagenta")
ggplot(data = tmp, aes(x = 1:length(ids), y = margin, color = SporfOutcome, label = ids)) + geom_point() + 
	geom_text(hjust = 0.01, nudge_x = 0.5, alpha = 0.4) + 
	geom_hline(yintercept = 0, color = 'red', lwd = 0.25) + xlab("") +
	scale_colour_manual(values = colorLegend) + 
	ggtitle("positive: sporf won,\nnegative: sporf lost")
```

![A scatter-plot showing the magnitide difference in losses between the two algorithms, ties at zero.](plotResults_Friendly_files/figure-html/unnamed-chunk-4-1.png)

<table>
 <thead>
  <tr>
   <th style="text-align:left;"> outliers_above </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> [11](https://openml.org/d/11) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> [1479](https://openml.org/d/1479) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> [1492](https://openml.org/d/1492) </td>
  </tr>
</tbody>
</table>



### SPORF Outcome:


```r
table(marDiff$margin >= 0)[2:1]
```

```
## 
##  TRUE FALSE 
##    36    13
```

```r
table(marDiff$margin > 0)[2:1]
```

```
## 
##  TRUE FALSE 
##    33    16
```


## Calculating a $p$-value

Using the parameters the yield the best outcome for each algorithm we
compute the difference $D_i = \min{RF} - \min{SPORF}$.  
Let $n' = n - (\text{number of ties})$, and $\bar{D_i} = \frac{1}{n'}\sum{\mathcal{I}_{D_i > 0}}$



```r
nprime <- length(ids) -  sum(tmp$Spo == 'tied')
q1 <- table(marDiff$margin > 0)[2]
pval <- 1 - pbinom(q = q1, size = nprime, prob = 0.5)
```

## The $p$-value is 8.2074567\times 10^{-4}



---


# Run time comparison:



```r
calcMins <- function(dat){ -1 * (diff(aggregate(dat[, .(run_time)], list(dat$clf), FUN = min)[, 2]))}

red <- data.table(Reduce(rbind, DAT))
#min_loss = min(test_loss), 
sumRuns <- 
	red[, .(min_loss = min(test_loss), sum_time = sum(run_time)), by = .(id,clf)][order(id, -clf)]

diffRuns <- 
	sumRuns[, .(diffLoss = diff(min_loss), diffTime = diff(sum_time)), by = .(id)]

ggplot(data = diffRuns, aes(x = diffLoss, y = diffTime, label = id)) + geom_point() + 
	geom_hline(yintercept = 0) + 
	geom_vline(xintercept = 0)
```

![](plotResults_Friendly_files/figure-html/unnamed-chunk-8-1.png)<!-- -->


## Contingency Table


```r
kable(
(g1 <- table(Time = diffRuns$diffTime < 0, Loss = diffRuns$diffLoss > 0))
)
```

<table>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> FALSE </th>
   <th style="text-align:right;"> TRUE </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:right;"> 14 </td>
   <td style="text-align:right;"> 7 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> TRUE </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 26 </td>
  </tr>
</tbody>
</table>

The diagonal reads `sklearn` won on time and loss 14 times (including ties), and `sporf` won on time and loss 26 times.


<!--
#   Time:
##  Working status:
### Comments:
####Soli Deo Gloria
--> 

