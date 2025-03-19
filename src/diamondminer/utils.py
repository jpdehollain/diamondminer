import matplotlib.pyplot as plt
import qcodes as qc
import numpy as np
import cv2

def get_data_from_qcodes_db(
        path_to_db: str, 
        run_id: int, 
        exp_name: str = "Initialization",
        plot: bool = False) -> list[np.array]:

    qc.dataset.initialise_or_create_database_at(path_to_db)
    dataset = qc.dataset.load_by_run_spec(experiment_name=exp_name, captured_run_id=run_id)
    if plot:
        qc.dataset.plot_dataset(dataset)
    df  = dataset.to_pandas_dataframe().reset_index()
    X_name, Y_name, Z_name = df.columns
    P_data, S_data = np.unique(df[X_name]), np.unique(df[Y_name])

    df_pivoted = df.pivot_table(values=Z_name, index=Y_name, columns=X_name)
    I_data = df_pivoted.to_numpy()
    return [P_data, S_data, I_data]

def extract_bias_point(
        lb_data: np.array,
        rb_data: np.array,
        current_data: np.array,
        minAngleDeg: float = -60,
        maxAngleDeg: float = -40,
        threshold: int = 50,
        minLineLength: int = 50,
        maxLineGap: int = 250,
        debug: bool = False,
        plot_results: bool = True) -> list[tuple]:

    lb_voltage_per_pixel = (lb_data[-1] - lb_data[0]) / len(lb_data)
    rb_voltage_per_pixel = (rb_data[-1] - rb_data[0]) / len(rb_data)

    Gx, Gy = np.gradient(current_data)
    G = (1/np.sqrt(2)) * np.sqrt(Gx**2 + Gy**2)
    G_log = np.log(np.abs(G))

    if debug:
        plt.title(r"$\log\vert G \vert$")
        plt.imshow(
            G_log,
            extent=[
                lb_data[0], lb_data[-1], rb_data[0], rb_data[-1]
            ],
            aspect='auto',
            origin='lower',
            cmap='binary')
        plt.xlabel("LB (V)")
        plt.ylabel("RB (V)")
        plt.show()

    G_log_normalized = G_log / G_log.max()
    G_uint = 255 * G_log_normalized
    image = G_uint.astype(np.uint8)

    se=cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    out_gray=cv2.divide(image, bg, scale=255)
    out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1] 

    edges = cv2.Canny(
            out_binary, 
            50, 
            150, 
            apertureSize=7,
            L2gradient=True
        )
    
    if debug:
        plt.title("Detected Edges")
        plt.imshow(
            edges,
            extent=[
                lb_data[0], lb_data[-1], rb_data[0], rb_data[-1]
            ],
            aspect='auto',
            origin='lower',
            cmap='binary')
        plt.xlabel("LB (V)")
        plt.ylabel("RB (V)")
        plt.show()

    lines = cv2.HoughLinesP(
        edges, # Input edge image
        1, # Distance resolution in pixels
        np.pi/180, # Angle resolution in radians
        threshold=threshold, # Min number of votes for valid line
        minLineLength=minLineLength, # Min allowed length of line
        maxLineGap=maxLineGap # Max allowed gap between line for joining them
    )

    filtered_lines = []
    if debug:
        plt.title("Detected Lines")
        plt.imshow(
            edges,
            aspect='auto',
            origin='lower',
            cmap='binary')
        plt.xlabel("LB (V)")
        plt.ylabel("RB (V)")
       
    for line in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2 = line[0]

        if x1 == x2:
            continue

        if debug:
            plt.plot([x1,x2], [y1, y2], c='r')


        slope = (y2 - y1) / (x2 - x1)
        slopeDeg = slope * 180 / np.pi

        if slopeDeg >= minAngleDeg and slopeDeg <= maxAngleDeg:
            filtered_lines.append(line[0])
        else:
            continue
    
    if debug:
        plt.show()
    
    bias_points = []
    for line in filtered_lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=line
        midpointX = 0.5 * (x1 + x2)
        midpointY = 0.5 * (y1 + y2)
        # Convert to voltage
        voltageX = lb_data[0] + midpointX * lb_voltage_per_pixel
        voltageY = rb_data[0] + midpointY * rb_voltage_per_pixel

        if plot_results:
            plt.title(r"$\log\vert I \vert$")
            plt.imshow(
                np.log(np.abs(current_data)),
                extent=[
                    lb_data[0], lb_data[-1], rb_data[0], rb_data[-1]
                ],
                aspect='auto',
                origin='lower',
                cmap='binary')
            plt.xlabel("LB (V)")
            plt.ylabel("RB (V)")
            plt.scatter([voltageX], [voltageY], c='w', marker="*", s=30)
            plt.text(voltageX, voltageY, s=f"({round(voltageX,3)},{round(voltageY,3)})", color='w')

        bias_points.append((round(voltageX,3), round(voltageY,3)))

    if plot_results or debug: 
        plt.show()
    
    return bias_points