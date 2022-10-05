using Serialization
using Images

train_images = deserialize("serialized_models/smaller_classification_667/viewable_images_train.data")


function save_all(images)
    # for (i, image) in enumerate(images)
    #     save("serialized_models/smaller_classification_667/train_images/$i.jpeg", image)
    # end

    # upscale by 4x (7x7 -> 28x28)
    scale = 8
    for (i, image) in enumerate(images)
        img2 = Matrix{Gray{Float64}}(undef, 7*scale, 7*scale)
        for i in 1:7*scale
            for j in 1:7*scale
                img2[i, j] = min.(max.(image[((i-1)÷scale)+1, ((j-1)÷scale)+1], 0.0), 1.0)
            end
        end
        save("serialized_models/smaller_classification_667/train_images/$i.png", img2)
    end
end

function save_all_grid(images)
    # there are 400 images
    # we want to make a 20x20 grid
    # each image is 7x7
    # so the images needs to be (20*7)x(20*7) = 140x140

    img = Matrix{Gray{Float64}}(undef, 140, 140)

    # julia indexing starts at 1

    for i in 1:20
        for j in 1:20
            img[(i-1)*7+1:i*7, (j-1)*7+1:j*7] = images[(i-1)*20 + j]
        end
    end

    # perform nearest neighbor upscale to 1400x1400
    # img = imresize(img, (1400, 1400))

    # can't use imresize because nearest neighbor is unavailable
    # so we'll do it manually using for loops

    img2 = Matrix{Gray{Float64}}(undef, 1400, 1400)
    for i in 1:1400
        for j in 1:1400
            img2[i, j] = img[((i-1)÷10)+1, ((j-1)÷10)+1]
        end
    end

    save("serialized_models/smaller_classification_667/train_images_grid.jpeg", img2)
end