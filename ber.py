import numpy as np

def ber_fixshreshold(prediction_mask, gt_mask, step_num=256, threshold = 127.5):
    predicted_mask = prediction_mask.copy()
    mask = gt_mask

    ber_final = 1
    ber_final_shadow = 1
    ber_finals = []
    ber_final_shadows = []


    mask[mask>=threshold] = 255
    mask[mask!=255] = 0

    predicted_mask[predicted_mask >= threshold] = 255
    predicted_mask[predicted_mask!=255] = 0

    total = predicted_mask/255 + mask/255
    zeros1 = np.zeros(np.shape(predicted_mask))
    zeros2 = np.zeros(np.shape(predicted_mask))
    zeros3 = np.zeros(np.shape(predicted_mask))
    zeros4 = np.zeros(np.shape(predicted_mask))
    ####[1,1]
    zeros1[total == 2] = 1
    ####[0,0]
    zeros2[total == 0] = 1
    ####[1,0]
    zeros3[(predicted_mask/255 - mask/255) == 1] = 1
    ####[0,1]
    zeros4[(predicted_mask/255 - mask/255) == -1] = 1

    TP = np.sum(zeros1)
    TN = np.sum(zeros2)
    FP = np.sum(zeros3)
    FN = np.sum(zeros4)
    # print('gggg', TP, TN, FP, FN, TP+TN+FP+FN, TP + FN, TN + FP)
    TP_shadow = np.sum(zeros1 * (mask/255))
    TN_shadow = np.sum(zeros2 * (mask/255))
    FP_shadow = np.sum(zeros3 * (mask/255))
    FN_shadow = np.sum(zeros4 * (mask/255))

    ber = 1 - 0.5*(TP/(TP + FN) + TN/(TN + FP))
    ber_shadow = 1 - (TP_shadow/(TP_shadow + FN_shadow))
    ber_finals.append(ber)
    ber_final_shadows.append(ber_shadow)
    return ber_finals, ber_final_shadows


def ber_ratioshreshold(prediction_mask, gt_mask, step_num=256):
    predicted_mask = prediction_mask.copy()
    mask = gt_mask

    ber_final = 1
    ber_final_shadow = 1
    ber_finals = []
    ber_final_shadows = []

    predict_pixels = (np.sort(np.reshape(predicted_mask,-1)))
    threshold_g_gt = np.mean(mask)
    mask[mask>=threshold_g_gt] = 255
    mask[mask!=255] = 0
    ratio = np.sum(mask/255)
    #threshold_g_pre = 5
    threshold_g_pre = predict_pixels[int(-ratio)]+1e-5

    predicted_mask[predicted_mask >= threshold_g_pre] = 255
    predicted_mask[predicted_mask!=255] = 0

    total = predicted_mask/255 + mask/255
    zeros1 = np.zeros(np.shape(predicted_mask))
    zeros2 = np.zeros(np.shape(predicted_mask))
    zeros3 = np.zeros(np.shape(predicted_mask))
    zeros4 = np.zeros(np.shape(predicted_mask))
    ####[1,1]
    zeros1[total == 2] = 1
    ####[0,0]
    zeros2[total == 0] = 1
    ####[1,0]
    zeros3[(predicted_mask/255 - mask/255) == 1] = 1
    ####[0,1]
    zeros4[(predicted_mask/255 - mask/255) == -1] = 1

    TP = np.sum(zeros1)
    TN = np.sum(zeros2)
    FP = np.sum(zeros3)
    FN = np.sum(zeros4)
    # print('gggg', TP, TN, FP, FN, TP+TN+FP+FN, TP + FN, TN + FP)
    TP_shadow = np.sum(zeros1 * (mask/255))
    TN_shadow = np.sum(zeros2 * (mask/255))
    FP_shadow = np.sum(zeros3 * (mask/255))
    FN_shadow = np.sum(zeros4 * (mask/255))

    ber = 1 - 0.5*(TP/(TP + FN) + TN/(TN + FP))
    ber_shadow = 1 - (TP_shadow/(TP_shadow + FN_shadow))
    ber_finals.append(ber)
    ber_final_shadows.append(ber_shadow)
    return ber_finals, ber_final_shadows
