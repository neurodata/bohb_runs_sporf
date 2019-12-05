

require(OpenML)



datasets = listOMLDataSets()  # returns active data sets



task = getOMLTask(task.id = 59L)
task


s <- getOMLStudy(99)

tmp <- sort(s$data$data.id)

write.table(tmp, file = "tasksCC18_R.dat", row.names = FALSE, col.names = FALSE)

opdat <- list()

i <- 1

for(si in tmp){
	print(si)

	opdat[[i]] <- getOMLDataSet(data.id = si)
	i <- i + 1
}


dims <- lapply(dat, function(x) dim(x$data))


li <- list()

rat <- opdat[[1]]$data

for(i in 1:nrow(rat)){
	li[[i]] <- 
		cbind(as.numeric(rat[i,])[-101])
}

A <- reshape2::melt(li)

tail(A)

A$Var2 <- NULL
sA <- A[A$L1 == 1, ]

ggplot(data = A, aes(x = Var1, y = value, group = L1, color = L1)) + geom_point()


