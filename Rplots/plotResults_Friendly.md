---
title: "BOHB Run on OpenML-100-Freindly"
author: "Jesse Leigh Patsolic"
output: 
  html_document:
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


## BOHB results as run on Synaptomes 1:


```r
dataDir <- "../output_syn1_100f/"

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

suppressWarnings({
md <- lapply(DAT, melt)
})

margin <- sapply(DAT, function(dat){ -1*diff(aggregate(dat[, .(loss)], list(dat$clf), FUN = min)[, 2]) })

marDiff <- data.table(margin, ids)
```

## A random sample of runs plotted with jittered violin plots


```r
p <- list()
j <- 1
for(i in sample(length(DAT), 9)){
#for(i in 1:length(DAT)){
	dat <- DAT[[i]]

	p[[j]] <- 
		ggplot(data = dat, 
					 aes(x = clf, y = loss, color = clf, group = clf)) + 
			  geom_violin(alpha = 0.3) + 
			  geom_jitter(size = max(dat$budget/max(dat$budget),0.1), height = 0) +
			  ggtitle(paste0("openml_d:", ids[i]))
	j <- j + 1
}

p$ncol = 3
do.call(grid.arrange, p)
```

![](plotResults_Friendly_files/figure-html/unnamed-chunk-1-1.png)<!-- -->


## A histogram showing the differnce in loss, i.e. ((loss(SKRF) - loss(SPORF))


```r
hist(marDiff$margin,breaks = "fre", 
	 main = "positive: sporf won,\n negative: sporf lost\n intervals are closed on the left, open on right", 
	 right = FALSE, xlab = "difference (RF - Sporf)")
abline(v = 0.0, col = 'red', lwd =2)
```

![](plotResults_Friendly_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

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

colorLegend <- c("red", "black", "darkgreen")
ggplot(data = tmp, aes(x = 1:length(ids), y = margin, color = SporfOutcome)) + geom_point() + 
	geom_hline(yintercept = 0, color = 'red', lwd = 0.25) + xlab("") +
	scale_colour_manual(values = colorLegend) + 
	ggtitle("positive: sporf won,\nnegative: sporf lost")
```

![](plotResults_Friendly_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

### SPORF Outcome:


```r
table(marDiff$margin >= 0)[2:1]
```

```
## 
##  TRUE FALSE 
##    37    12
```

```r
table(marDiff$margin > 0)[2:1]
```

```
## 
##  TRUE FALSE 
##    29    20
```

---





<!--
#   Time:
##  Working status:
### Comments:
####Soli Deo Gloria
--> 

