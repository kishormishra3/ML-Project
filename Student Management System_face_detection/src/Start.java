package front;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.LineBorder;

public class Start implements ActionListener,MouseListener{
	JFrame frame;
	JLabel ll;
	JTextField user,pass;
	Start(){
	frame=new JFrame("Student Login");
	frame.getContentPane().setBackground( Color.BLACK );
	frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	frame.getContentPane().getSize();
	ll = new JLabel();
	ImageIcon imageIcon = new ImageIcon(new ImageIcon("F:\\g.jpg").getImage().getScaledInstance(100, 100, Image.SCALE_DEFAULT));
	ll.setIcon(imageIcon);
	JPanel panel2 =new JPanel(new FlowLayout());
	JPanel panel1 =new JPanel(new GridLayout(2,2));
	JPanel panel3 =new JPanel(new FlowLayout());;  
	JLabel l3=new JLabel("<html><span style='font-size:20px;'>"+"<b><i>Welcome to Student Management System<i><b>"+"</span></html>");
	l3.setFont(new Font("Serif", Font.BOLD, 14));
	
	Button b1= new Button("Submit");
	b1.addActionListener(this);
	JLabel l1=new JLabel("<html><span style='font-size:15px;'>"+"<i>Username: <i>"+"</span></html>");
	//l1.setForeground(Color.WHITE);
	user =new JTextField("username",20);
	user.setBorder(new LineBorder(Color.WHITE, 15));
	user.setSize(20,20);
	JLabel l2=new JLabel("<html><span style='font-size:15px;'>"+"<i>Password: <i>"+"</span></html>");
	//l2.setForeground(Color.WHITE);

	l1.setFont(new Font("Serif", Font.BOLD, 14));
	l2.setFont(new Font("Serif", Font.BOLD, 14));
	pass =new JTextField("password",20);
	//panel1.setBounds(10,10,10,10);
	pass.setBorder(new LineBorder(Color.WHITE, 15));
	panel2.add(l3,SwingConstants.CENTER);
	panel2.add(ll);
	panel1.add(l1);
	


	panel1.add(user);
	panel1.add(l2);
	panel1.add(pass);
	panel3.add(b1);
	  l1.setHorizontalAlignment(JLabel.CENTER);
	  l2.setHorizontalAlignment(JLabel.CENTER);
	  panel1.setBackground(Color.WHITE);
	  panel2.setBackground(Color.WHITE);
	  panel3.setBackground(Color.WHITE);
	  b1.addActionListener (this);
	frame.add(panel2);
	frame.setResizable(false);
	frame.add(panel1);
	frame.add(panel3);
	frame.setLocation(200, 100);
	
	frame.setLayout(new GridLayout(3,1));
	frame.setSize(550,450);
	frame.setVisible(true);
	}
	public void actionPerformed(ActionEvent e) {
		String u=user.getText();
		String p=pass.getText();
		if(u.equalsIgnoreCase("kishor"))
		{
			if(p.equalsIgnoreCase("kishor"))
			{
				frame.setVisible(false);
				new Option();
			}
		}
		else
		{
			JOptionPane.showMessageDialog(null, "Error in Username or Password");
		}	
	}
	@Override
	public void mouseClicked(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void mouseEntered(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void mouseExited(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void mousePressed(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void mouseReleased(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
}
