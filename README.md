## Chuẩn bị data
### Tiền xử lý
- Chỉ sử dụng title của các bài báo
- Các title sử dụng VnCoreNLP để tách từ
- Đánh số các từ để chuyển thành vector
- Không sử dụng pretrain embedding
### Cách tạo mẫu train
Với mỗi user có số lượng item clicked lớn hơn tham số MAX_SENTS thì chia thành các mẫu nhỏ theo dạng như sau:

Ví dụ với tham số MAX_SENTS là 2, một user có 4 clicked items là 1 -> 4, sẽ tạo được 2 mẫu train:
- Browsed Items gồm items 1, 2 -> positive candidate item là 3, negative positive lựa chọn ngẫu nhiên khác 4 clicked items
- Browsed Items gồm items 2, 3 -> positive candidate item là 4, negative positive lựa chọn ngẫu nhiên khác 4 clicked items

Với mỗi mẫu train thì chỉ có 1 browsed_items, còn lại là negative.

## Mô hình
Mô hình được chia thành 2 phần chính là: 
- encoder để tính embedding cho các text
- user_presentation để tính browsed_items embedding

2 phần được chia vào 2 hàm forward_encoder và forward_browsed_news trong lớp AttentiveMultiView

## Cách huấn luyện
Với mỗi mẫu train:
- Browsed_items bao gồm các title user đã đọc: với mỗi items đưa qua hàm forward_encoder để tính embedding cho text.
Sau đó concate các embedding lại rồi đưa qua lớp attention -> user_embedding có kích thước (1, 400)
- Candidate items lần lượt được đưa qua hàm forward_encoder tính embedding cho text -> kích thước (npratio+1, 400)
- Nhân user_embedding với candidate_embedding lại với nhau được 1 vector kích thước (npratio + 1)
- đưa ma trận kết quả qua softmax và tính loss với nhãn, các negative candidate nhãn 0, positive candidate nhãn 1.

## Cách eval model

- Cho tập transaction test của người dùng. Xử lý giống với các tạo mẫu train.

- Các candidate là toàn bộ các bài báo có trong dataset. Tính tất cả embedding của các bài báo. Ví dụ có 1000 bài báo
thì vector_candidate có kích thước (1000, 400)

- Với mỗi mẫu dữ liệu, sử dụng browsed_items để tính user_embedding có kích thước (1, 400) sau đó nhân ma trận với vector_candidate,
ranking kết quả tìm ra 10 bài có điểm cao nhất, sau đó kiểm tra nếu positive candidate có trong 10 bài đó thì tính là đúng, ngược lại sai.
Xét như vậy với toàn bộ user để đánh giá MRR@10

## Cách dự đoán cho product
Các candidate toàn toàn bộ bài báo có trong database. Cũng tính tất cả embedding của các bài báo trong database.

Fit browsed_items để tìm user_embedding, sau đó nhân ma trận tương tự eval, dự đoán ra 10 items có ranking cao nhất.