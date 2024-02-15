#.libPaths( c(.libPaths(), "~/R_libs/"))

suppressMessages(library(BandNorm))

args <- commandArgs(trailingOnly = TRUE)
interaction_dir <- as.character(args[1])
genes <- as.character(args[2])
out_file <- as.character(args[3])
depth_norm <- as.logical(args[4])

geneObj = hg19Annotations

if ( genes == "mm10") {
  geneObj <- mm10Annotations
} else if ( genes == "hg19") {
  geneObj <- hg19Annotations
}


gad_score = scGAD(path = interaction_dir, genes = geneObj, depthNorm = depth_norm, cores = 8)

write.table(gad_score, file = out_file, row.names=TRUE, sep="\t")

