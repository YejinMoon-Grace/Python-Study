import pydicom  # DICOM 파일을 읽고 태그 및 픽셀 데이터를 추출하기 위한 라이브러리
import matplotlib.pyplot as plt  # 영상 데이터를 시각화하여 PNG 형식으로 렌더링
import numpy as np  # 픽셀 데이터 정규화 및 배열 처리
from tkinter import Tk, Button, Label, filedialog, Text, Scrollbar, Frame  # GUI 창과 위젯 생성
from PIL import Image, ImageTk  # 렌더링된 영상을 GUI에 표시하기 위한 Pillow 모듈
import io  # 메모리 버퍼를 사용하여 이미지 데이터를 처리
import os  # 파일 경로 및 이름을 처리


class DicomViewer:
    def __init__(self, root):
        """
        DicomViewer 클래스의 초기화 메서드.
        GUI 창과 위젯(라벨, 버튼, 이미지 창, 텍스트 창, 스크롤바)을 생성하고 초기 설정을 수행합니다.

        Args:
            root (Tk): tkinter의 루트 창 객체.
        """
        self.root = root  # 루트 창 객체 저장
        self.root.title("DICOM Viewer with Fixed Textbox")  # 창 제목 설정
        self.root.geometry("800x600")  # 창 크기를 800x600 픽셀로 설정

        # GUI 요소 생성 및 배치
        # 파일 로드 상태를 표시하는 상단 라벨 (예: "DICOM 파일을 선택하세요")
        self.label = Label(self.root, text="DICOM 파일을 선택하세요", font=("Arial", 12))
        self.label.pack(pady=10)  # 상단에 배치, 세로 여백 10픽셀

        # 영상 크기(Rows x Columns)를 표시하는 라벨, 초기값은 "N/A"
        self.size_label = Label(self.root, text="Image Size: N/A", font=("Arial", 10))
        self.size_label.pack(pady=5)  # 상단 라벨 아래 배치, 세로 여백 5픽셀

        # DICOM 파일 선택을 위한 버튼
        self.select_button = Button(self.root, text="파일 선택", font=("Arial", 10), command=self.load_dicom)
        self.select_button.pack(pady=5)  # 크기 라벨 아래 배치, 세로 여백 5픽셀

        # DICOM 영상을 표시할 이미지 라벨
        self.image_label = Label(self.root)
        self.image_label.pack(pady=10, expand=True)  # 중앙에 배치, 창 크기 조정 시 확장 가능

        # 태그 목록을 표시하기 위한 프레임 (하단 고정)
        self.text_frame = Frame(self.root)
        self.text_frame.pack(side="bottom", fill="x", pady=10)  # 하단에 고정, 수평으로 채움

        # 스크롤바 생성 및 배치
        self.scrollbar = Scrollbar(self.text_frame, width=20)  # 스크롤바 너비 20픽셀로 설정
        self.scrollbar.pack(side="right", fill="y")  # 프레임 오른쪽에 배치, 세로로 채움

        # 태그 목록을 표시할 텍스트 창
        self.tag_text = Text(self.text_frame, height=10, wrap="word", yscrollcommand=self.scrollbar.set,
                             font=("Courier", 10))
        self.tag_text.pack(side="left", fill="x", expand=True)  # 프레임 왼쪽에 배치, 수평으로 확장
        self.scrollbar.config(command=self.tag_text.yview)  # 스크롤바와 텍스트 창 연결
        self.tag_text.insert("end", "DICOM 태그 정보가 여기에 표시됩니다.\n스크롤하여 태그를 확인하세요.")  # 초기 메시지 삽입
        self.tag_text.config(state="disabled")  # 텍스트 창을 읽기 전용으로 설정

        # 마우스 휠 스크롤 이벤트 바인딩 (플랫폼별 호환성 확보)
        self.tag_text.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows: 마우스 휠 이벤트
        self.tag_text.bind("<Button-4>", self.on_mouse_wheel)  # Linux: 마우스 휠 업
        self.tag_text.bind("<Button-5>", self.on_mouse_wheel)  # Linux: 마우스 휠 다운

    def on_mouse_wheel(self, event):
        """
        마우스 휠 스크롤 이벤트를 처리하는 메서드.
        텍스트 창에서 마우스 휠을 사용하여 태그 목록을 스크롤합니다.

        Args:
            event: 마우스 휠 이벤트 객체 (delta: Windows, num: Linux).

        Returns:
            str: "break"를 반환하여 이벤트 전파를 중단.
        """
        if event.delta:  # Windows: delta 값으로 스크롤 방향 결정
            self.tag_text.yview_scroll(-1 * (event.delta // 120), "units")
        elif event.num == 4:  # Linux: 마우스 휠 업
            self.tag_text.yview_scroll(-1, "units")
        elif event.num == 5:  # Linux: 마우스 휠 다운
            self.tag_text.yview_scroll(1, "units")
        return "break"  # 이벤트 전파 중단

    def load_dicom(self):
        """
        DICOM 파일을 로드하고 영상을 표시하며 태그 정보를 출력하는 메서드.
        사용자가 선택한 DICOM 파일을 읽어 영상 크기, 영상, 태그 정보를 GUI에 표시합니다.
        """
        try:
            # 파일 선택 대화상자를 열어 DICOM 파일 선택
            file_path = filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm")])
            if not file_path:  # 파일 선택이 취소된 경우
                return

            # pydicom으로 DICOM 파일 읽기
            dicom_data = pydicom.dcmread(file_path)
            pixel_array = dicom_data.pixel_array  # 픽셀 데이터 추출 (2D 배열)

            # 영상 크기 추출 (Rows와 Columns 태그)
            rows = dicom_data.get((0x0028, 0x0010), None)  # Rows 태그 (행 수)
            cols = dicom_data.get((0x0028, 0x0011), None)  # Columns 태그 (열 수)
            size_text = f"Image Size: {rows.value if rows else 'N/A'} x {cols.value if cols else 'N/A'}"
            self.size_label.configure(text=size_text)  # 영상 크기 라벨 업데이트

            # 픽셀 데이터를 0-255 범위로 정규화 (uint8 타입으로 변환)
            pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
            pixel_array = pixel_array.astype(np.uint8)

            # Matplotlib로 이미지 생성 (흑백 영상)
            plt.figure(figsize=(3, 3))  # 300x300 픽셀에 맞게 캔버스 크기 설정 (1인치=100픽셀)
            plt.imshow(pixel_array, cmap='gray')  # 흑백 컬러맵 사용
            plt.axis('off')  # 축 숨김

            # 이미지를 메모리 버퍼에 저장
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')  # PNG 형식으로 저장
            plt.close()  # Matplotlib 창 닫기
            buf.seek(0)  # 버퍼 포인터를 처음으로 이동

            # PIL을 사용해 tkinter에 표시
            image = Image.open(buf)  # 버퍼에서 이미지 열기
            image = image.resize((300, 300), Image.LANCZOS)  # 300x300 픽셀로 크기 조정 (LANCZOS 필터 사용)
            photo = ImageTk.PhotoImage(image)  # tkinter에서 사용할 이미지 객체 생성
            self.image_label.configure(image=photo)  # 이미지 라벨에 표시
            self.image_label.image = photo  # 참조 유지 (가비지 컬렉션 방지)

            # 파일 이름 표시
            self.label.configure(text=f"Loaded: {os.path.basename(file_path)}")

            # DICOM 태그 정보 표시
            self.display_dicom_tags(dicom_data)

        except Exception as e:
            # 오류 발생 시 처리
            self.label.configure(text=f"오류 발생: {str(e)}")  # 상단 라벨에 오류 메시지 표시
            self.size_label.configure(text="Image Size: N/A")  # 영상 크기 초기화
            self.tag_text.config(state="normal")  # 텍스트 창 편집 가능 상태로 전환
            self.tag_text.delete("1.0", "end")  # 기존 텍스트 삭제
            self.tag_text.insert("end", f"⚠ 태그 로드 오류: {str(e)}\n")  # 오류 메시지 삽입
            self.tag_text.config(state="disabled")  # 텍스트 창 다시 읽기 전용으로 설정

    def display_dicom_tags(self, dicom_data):
        """
        DICOM 태그 정보를 텍스트 창에 포맷팅하여 출력하는 메서드.
        모든 태그를 순회하며 태그 ID, 이름, 값을 표 형식으로 표시합니다.

        Args:
            dicom_data (pydicom.Dataset): 읽어들인 DICOM 데이터셋 객체.
        """
        self.tag_text.config(state="normal")  # 텍스트 창 편집 가능 상태로 전환
        self.tag_text.delete("1.0", "end")  # 기존 텍스트 삭제

        # 태그 정보를 포맷팅하여 문자열로 생성
        tag_info = "DICOM 태그 목록:\n\n"
        for elem in dicom_data:  # 모든 태그 순회
            try:
                # 태그 ID, 이름, 값을 포맷팅 (값은 50자 제한으로 잘림 방지)
                tag_info += f"{elem.tag} | {elem.name:<40} | {str(elem.value)[:50]:<50}\n"
            except Exception as e:
                # 태그 디코딩 오류 시 오류 메시지 포함
                tag_info += f"{elem.tag} | {'Error decoding':<40} | {str(e)[:50]:<50}\n"

        self.tag_text.insert("end", tag_info)  # 포맷팅된 태그 정보 삽입
        self.tag_text.config(state="disabled")  # 텍스트 창 읽기 전용으로 설정


if __name__ == "__main__":
    # 프로그램 진입점: tkinter 루트 창 생성 및 DicomViewer 인스턴스 실행
    root = Tk()
    app = DicomViewer(root)
    root.mainloop()  # GUI 이벤트 루프 시작

