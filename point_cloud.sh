workdir=$1
export CUDA_VISIBLE_DEVICES=0

for time_frame in $(ls $workdir/frames); do
    time_frame_path="$workdir/frames/$time_frame"
    input_path="$workdir/sparse/0"
    output_path="$workdir/colmap/$time_frame"
    rm -rf $output_path
    mkdir -p $output_path

    database_path="$output_path/database.db"
    
    colmap feature_extractor --database_path $database_path \
        --image_path $time_frame_path \
        --ImageReader.single_camera 1

    colmap exhaustive_matcher --database_path $database_path

    mkdir -p $output_path/sparse
    colmap point_triangulator \
        --database_path $database_path \
        --image_path $time_frame_path \
        --input_path $input_path \
        --output_path $output_path/sparse \
        --Mapper.ba_global_function_tolerance 0.000001
done