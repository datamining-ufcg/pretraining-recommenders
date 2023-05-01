for target_folder in 1m 100k
do
    for folder in 1m 10m 20m 25m
    do
        Rscript transfer_plots.R "results/target_$target_folder/leakage_user_$folder" "$target_folder"
    done

done

Rscript transfer_plots.R "results/target_100k/ml100k" "100k"

Rscript extra_plots.R
# Rscript paper_plots.R "results/netflix"

rm Rplots.pdf