paste samples.txt populations.txt | awk 'BEGIN{print "##fileformat=VCFv4.2\n##contig=<ID=1>\n##contig=<ID=2>\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Phased Genotype\">"}{printf "##SAMPLE=<ID=%s,Population=%s>\n",$1,$2}END{printf "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"; while((getline<"samples.txt")>0) printf "\t%s",$1; printf "\n"}' > header.vcf

awk -v N=$(wc -l < samples.txt) '
BEGIN{pos=0}
{
  # count alleles at this site
  delete cnt
  for(i=1;i<=2*N;i++) cnt[substr($4,i,1)]++

  # choose REF = major allele, ALT = minor allele
  ref=""; alt=""
  for(a in cnt){
    if(ref=="" || cnt[a] > cnt[ref]){ alt=ref; ref=a }
    else if(alt=="" || cnt[a] > cnt[alt]) alt=a
  }

  # col3-1 monomorphic reference sites
  for(i=1;i<$3;i++){
    pos++
    printf "%s\t%d\t.\t%s\t%s\t.\tPASS\t.\tGT", $1, pos, ref, alt
    for(j=1;j<=N;j++) printf "\t0|0"
    printf "\n"
  }

  # polymorphic site (PHASED)
  pos++
  printf "%s\t%d\t.\t%s\t%s\t.\tPASS\t.\tGT", $1, pos, ref, alt
  for(j=1;j<=N;j++){
    hapA = substr($4, j,   1)   # first haplotype
    hapB = substr($4, j+N, 1)   # second haplotype
    gt = (hapA==ref?0:1) "|" (hapB==ref?0:1)
    printf "\t%s", gt
  }
  printf "\n"
}' msmc_input_1.txt > body.vcf



cat header.vcf body.vcf > new_wc_1.vcf

